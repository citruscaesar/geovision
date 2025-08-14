from typing import Literal, Optional
from numpy.typing import NDArray

import os 
import h5py
import torch
import shutil
import subprocess
import numpy as np
import pandas as pd
import pandera as pa
import rasterio as rio
import geopandas as gpd
import imageio.v3 as iio

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from shapely.geometry import shape
from torchvision.transforms import v2 as T

from geovision.data import Dataset, DatasetConfig
from geovision.io.local import FileSystemIO as fs

class RAMP:
    local = Path.home() / "datasets" / "ramp"
    class_names = ("background", "building") 
    default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="stratified",
        tabular_sampler_params={"split_on": "location", "val_frac": 0.15, "test_frac": 0.15},
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        target_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]),
        train_aug=T.Compose([T.RandomCrop(128, pad_if_needed=True), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
        eval_aug=T.RandomResizedCrop(128)
    )

    @staticmethod
    def extract(cls):
        assert shutil.which('s5cmd') is not None, "couldn't find s5cmd on the system"
        os.environ["S3_ENDPOINT_URL"] = "https://data.source.coop" 
        subprocess.run(["s5cmd", "--no-sign-request", "sync", "s3://ramp/ramp", str(fs.get_new_dir(cls.local, "staging"))])
    
    @staticmethod
    def download(cls):
        assert shutil.which('s5cmd') is not None, "couldn't find s5cmd on the system"
        os.environ["S3_ENDPOINT_URL"] = "https://sin1.contabostorage.com" 
        subprocess.run(["s5cmd", "--no-sign-request", "sync", "s3://ramp/hdf5/ramp.h5", str(fs.get_new_dir(cls.local, "hdf5"))])

    @classmethod
    def load(
        cls, 
        table: Literal["index", "spatial"], 
        src: Literal["staging", "imagefolder", "hdf5"],
        subset: Literal["ramp"] = "ramp"
    ) -> pd.DataFrame:

        assert src in ("staging", "imagefolder", "hdf5")
        assert subset == "ramp" 

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/f"{subset}.h5", key=table, mode="r")

        if src == "imagefolder":
            imagefolder_path = fs.get_valid_dir_err(cls.local/"imagefolder", empty_ok=False)
            try:
                return pd.read_hdf(imagefolder_path/"metadata.h5", key=table, mode="r")
            except OSError:
                assert table in ("index", "spatial")

                image_paths = sorted((imagefolder_path/"images").rglob("*.tif"))
                if len(image_paths) == 0:
                    image_paths = sorted((imagefolder_path/"images").rglob("*.jpg"))

                index_df = (
                    pd.DataFrame({"image_path": image_paths})
                    .assign(image_path = lambda df: df["image_path"].apply(lambda x: '/'.join(x.split('/')[-3:])))
                    .assign(mask_path = lambda df: df["image_path"].str.replace("images", "masks"))
                )
                spatial_df = (
                    index_df[["image_path"]]
                    .assign(image_height = 256)
                    .assign(image_width = 256)
                )

                index_df.to_hdf(imagefolder_path/"metadata.h5", key="index", mode="w", format="fixed", complib="zlib", complevel=9)
                spatial_df.to_hdf(imagefolder_path/"metadata.h5", key="spatial", mode="a", format="fixed", complib="zlib", complevel=9)

                if table == "index":
                    return index_df
                elif table == "spatial":
                    return spatial_df
        
        elif src == "staging":
            staging_path = fs.get_valid_dir_err(cls.local, "staging")
            try:
                #raise OSError
                return pd.read_hdf(staging_path/"metadata.h5", key = table, mode = "r")
            except OSError:
                assert table in ("index", "spatial")
                
                index_dfs = list()
                spatial_tables = {k:list() for k in ("image_path", "crs", "transform")} 

                for label_path in tqdm(sorted(staging_path.rglob("*.geojson"))):

                    # get spatial information of the corresponding raster image, as the label geometry needs to be reprojected to this later 
                    image_path = '/'.join(label_path.as_posix().split('/')[-3:]).replace("labels", "source").replace(".geojson", ".tif")
                    with rio.open(staging_path/image_path) as raster:
                        spatial_tables["image_path"].append(image_path)
                        spatial_tables["crs"].append(raster.crs.to_epsg())
                        spatial_tables["transform"].append(raster.transform.to_gdal())
 
                    index_dfs.append(
                        gpd.read_file(label_path).to_wkt() # read vector file w/geometry as str (or to binary using to_wkb(); else can't save to hdf5) 
                        .assign(image_path = image_path) # :image_path = location of filesystem (local/staging/...) of the corresponding raster image
                    )
           
                index_df = pd.concat(index_dfs).set_index("image_path").sort_index()
                spatial_df = pd.DataFrame(spatial_tables).set_index("image_path").sort_index()

                index_df.to_hdf(staging_path/"metadata.h5", key="index", mode="w", format="fixed", complib="zlib", complevel=9)
                spatial_df.to_hdf(staging_path/"metadata.h5", key="spatial", mode="a", format="fixed", complib="zlib", complevel=9)

                if table == "index":
                    return index_df
                elif table == "spatial":
                    return spatial_df

    @classmethod
    def transform(cls, to: Literal["imagefolder", "hdf5"], to_jpg: bool = False, jpg_quality: int = 95, chunk_size: Optional[int] = None):
        assert to in ("imagefolder", "hdf5")
        if to_jpg:
            assert isinstance(jpg_quality, int) and jpg_quality <= 95 and jpg_quality > 0
        if chunk_size is not None:
            assert isinstance(chunk_size, int)

        def _get_imagefolder_name(staging_name: str) -> str:
            loc, _, name = staging_name.split("/")
            return f"{loc.removeprefix("ramp_")}/{name}"
        
        def _get_rasterized_mask(mask_gdf: gpd.GeoDataFrame | gpd.GeoSeries, epsg:int, transform: rio.Affine) -> NDArray:
            return rio.features.rasterize(
                shapes = [(shape(g), 255) for g in mask_gdf.to_crs(epsg=epsg).geometry if g is not None], 
                out_shape = (256, 256), 
                transform = transform, 
                dtype = np.uint8
            )
            
        staging_path = fs.get_valid_dir_err(cls.local, "staging") 
        geometry_df = (
            gpd.GeoDataFrame(cls.load("index", "staging", "ramp"))
            .assign(geometry = lambda df: gpd.GeoSeries.from_wkt(df["geometry"]))
            .set_geometry("geometry") 
            .set_crs(epsg=4326)
        )
        index_df = (
            pd.DataFrame()
            .assign(image_src = geometry_df.index.unique())
            .assign(image_path = lambda df: df["image_src"].apply(_get_imagefolder_name))
        )
        spatial_df = cls.load("spatial", "staging", "ramp")
        spatial_df.index = spatial_df.index.map(_get_imagefolder_name)

        if to == "imagefolder":
            imagefolder_path = fs.get_new_dir(cls.local, "imagefolder")

            # Move / Copy Images
            if to_jpg:
                index_df["image_path"] = index_df["image_path"].str.replace(".tif", ".jpg")
                spatial_df.index = spatial_df.index.str.replace(".tif", ".jpg")

                for _, row in tqdm(index_df.iterrows(), total = len(index_df), desc = f"re-encoding images to .jpg (quality={jpg_quality})..."):
                    src, dst = staging_path/row["image_src"], imagefolder_path/row["image_path"]
                    if not dst.parent.is_dir():
                        dst.parent.mkdir(parents=True)
                    iio.imwrite(dst, iio.imread(src, extension=".tif"), extension=".jpg", quality = jpg_quality)

            else:
                print("moving images...", end = " ")
                for dir_path in staging_path.iterdir():
                    if dir_path.is_dir() and dir_path.name.startswith("ramp_"):
                        shutil.copytree(dir_path/"source", imagefolder_path/"images"/dir_path.name.removeprefix("ramp_"), dirs_exist_ok=True)
                print("done")

            index_df = index_df.drop(columns=["image_src"]) # since all source images have been moved
            geometry_df.index = geometry_df.index.map(_get_imagefolder_name)

                #######-------------------------TODO: Wrap this in a function and use multiprocessing to speed it up----------------------------######
            for _, row in tqdm(index_df.iterrows(), total = len(index_df), desc = "rasterizing labels..."):

                # get spatial metadata
                raster_georef_row, vector_geometry_table = spatial_df.loc[row["image_path"]], geometry_df.loc[[row["image_path"]]]
                epsg, transform = raster_georef_row["crs"], rio.Affine.from_gdal(*raster_georef_row["transform"])

                # create mask raster, NOTE: _get_rasterized_mask() will have to be moved out to a @staticmethod for mpl to work
                mask = _get_rasterized_mask(vector_geometry_table, epsg, transform)

                # save mask to fs 
                mask_path = imagefolder_path/"masks"/row["image_path"]
                if not mask_path.parent.is_dir():
                    mask_path.parent.mkdir(parents = True)
                if to_jpg:
                    iio.imwrite(mask_path, mask, extension=".jpg", quality = jpg_quality)
                else:
                    with rio.open(mask_path, 'w', "GTiff", 256, 256, 1, rio.crs.CRS.from_epsg(epsg), transform, np.uint8) as raster:
                        raster.write(mask, 1)
                #######-------------------------------------------------------------------------------------------------------------------------######
            
            index_df["mask_path"] = index_df["image_path"]
            index_df["location"] = index_df["image_path"].apply(lambda x: x.split('/')[0])
            index_df.to_hdf(imagefolder_path/"metadata.h5", key="index", mode="w", format="fixed", complib="zlib", complevel=9)

            spatial_df["image_height"] = 256
            spatial_df["image_width"] = 256
            spatial_df.reset_index().to_hdf(imagefolder_path/"metadata.h5", key="spatial", mode="r+", format="fixed", complib="zlib", complevel=9)

        elif to == "hdf5":
            hdf5_path = fs.get_new_dir(cls.local, "hdf5") / "ramp.h5" 

            performance_kwargs = dict() 
            if chunk_size is not None:
                performance_kwargs["chunks"] = chunk_size

            with h5py.File(hdf5_path, mode = "a") as f:
                if f.get("masks") is not None:
                    del f["masks"]

                masks = f.create_dataset("masks", shape = len(index_df), dtype = h5py.special_dtype(vlen=np.uint16), **performance_kwargs)

                if to_jpg:
                    index_df["image_path"] = index_df["image_path"].str.replace(".tif", ".jpg")
                    spatial_df.index = spatial_df.index.str.replace(".tif", ".jpg")
                    images = f.create_dataset("images", shape = len(index_df), dtype = h5py.special_dtype(vlen=np.uint8), **performance_kwargs)
                else:
                    if chunk_size is not None:
                        performance_kwargs["chunks"] = (chunk_size, 256, 256, 3)
                    images = f.create_dataset("images", shape = (len(index_df), 256, 256, 3), **performance_kwargs)

                for idx, row in tqdm(index_df.iterrows(), total = len(index_df), desc="saving images and labels to hdf5..."):
                    # load raster and get georegistration info
                    with rio.open(staging_path/row["image_src"]) as raster:
                        image = raster.read().transpose(1,2,0)
                        epsg, transform = raster.crs.to_epsg(), raster.transform
                    # save image
                    if to_jpg:
                        images[idx] = np.frombuffer(iio.imwrite('<bytes>', image, extension=".jpg", quality=jpg_quality), dtype=np.uint8)
                    else:
                        images[idx] = image
                    # get vector table from geometry df, generate raster mask and save as run-length encoded sequence 
                    masks[idx] = cls.rle_encode(_get_rasterized_mask(geometry_df.loc[[row["image_src"]]], epsg, transform))

                    # mask = _get_rasterized_mask(geometry_df.loc[[row["image_src"]]], epsg, transform)
                    # print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean(), mask.std())
                    # mask = cls.rle_encode(mask)
                    # print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean(), mask.std())
                    # masks[idx] = mask 

            index_df = index_df.drop(columns=["image_src"])
            index_df["mask_path"] = index_df["image_path"]
            index_df["location"] = index_df["image_path"].apply(lambda x: x.split('/')[0])
            index_df.to_hdf(hdf5_path, key="index", mode='r+', format="fixed", complib="zlib", complevel=9)
            spatial_df["image_height"] = 256
            spatial_df["image_width"] = 256
            spatial_df.reset_index().to_hdf(hdf5_path, key="spatial", mode='r+', format="fixed", complib="zlib", complevel=9)

    @staticmethod
    def rle_encode(img):
        """
        Encodes a binary image using run-length encoding (RLE).
        
        Args:
            img (numpy.ndarray): A 2D uint8 binary image with values 0 and 255.

        Returns:
            numpy.ndarray: A 1D array of RLE pairs (start, length).
        """
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect edges
        changes = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Find transition points

        starts = changes[::2]  # Start indices
        lengths = changes[1::2] - starts  # Compute lengths

        if len(lengths) < len(starts):  # Handle edge case where last start has no matching end
            lengths = np.append(lengths, len(pixels) - starts[-1])

        return np.vstack((starts, lengths)).T.flatten().astype(np.uint16)

    @staticmethod
    def rle_decode(rle, shape):
        """
        Decodes a run-length encoded binary image.
        
        Args:
            rle (numpy.ndarray): A 1D array of RLE pairs (start, length).
            shape (tuple): The shape (height, width) of the original image.

        Returns:
            numpy.ndarray: A 2D uint8 binary image with values 0 and 255.
        """
        img = np.zeros(shape[0] * shape[1], dtype=np.uint16)
        rle = rle.reshape(-1, 2)  # Ensure it is in (start, length) pairs

        for start, length in rle:
            img[start : start + length] = 255  # Set foreground pixels to 255

        return img.reshape(shape)

RAMPIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "mask_path": pa.Column(str, coerce=True),
        "location": pa.Column(str, coerce=True),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    },
    index=pa.Index(int)
)

class RAMP_Building_Segmentation_Imagefolder(Dataset):
    name = "ramp"
    task = "segmentation"
    subtask = "semantic"
    storage = "imagefolder"
    class_names = ("background", "building") 
    num_classes = 2 
    root = RAMP.local/"imagefolder"
    schema = RAMPIndexSchema 
    config = RAMP.default_config
    loader = RAMP.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=True)
        self.identity_matrix = np.eye(self.num_classes, dtype=np.uint8)
        self.mask_shape = (256, 256) 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        image = iio.imread(idx_row["image_path"]) # image.shape = (H, W, num_channels)
        mask = iio.imread(idx_row["mask_path"]).squeeze() # mask.shape = (H, W)
        mask = self.identity_matrix[np.clip(mask, 0, 1)] # mask.shape = (H, W, num_classes)
        image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        return image, mask, idx_row["df_idx"]

class RAMP_Building_Segmentation_HDF5(Dataset):
    name = "ramp"
    task = "segmentation"
    subtask = "semantic"
    storage = "hdf5"
    class_names = ("background", "building") 
    num_classes = 2 
    root = RAMP.local/"hdf5"/"ramp.h5"
    schema = RAMPIndexSchema 
    config = RAMP.default_config
    loader = RAMP.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=False)
        self.identity_matrix = np.eye(self.num_classes, dtype = np.uint8)
        self.mask_shape = (256, 256) 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode = "r") as f:
            image = f["images"][idx_row["df_idx"]]
            mask = RAMP.rle_decode(f["masks"][idx_row["df_idx"]], shape = self.mask_shape) # mask: (H, W)

        if image.ndim != 3:
            image = iio.imread(BytesIO(image), extension=".jpg")

        # _, ax = plt.subplots(1, 1)
        # ax.imshow(image.astype(np.uint8))
        # ax.imshow(mask, alpha = 0.4, cmap = "Reds")
        # print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        # print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())

        mask = self.identity_matrix[np.clip(mask, 0, 1)] # mask: (H, W, num_channels)
        image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        return image, mask, idx_row["df_idx"]