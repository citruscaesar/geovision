from typing import Literal, Optional, Callable, Sequence
from numpy.typing import NDArray

import h5py
import py7zr
import torch
import shutil
import zipfile
import subprocess
import multivolumefile

import numpy as np
import pandas as pd
import pandera as pa
import rasterio as rio
import geopandas as gpd
import imageio.v3 as iio
import torchvision.transforms.v2 as T

from tqdm import tqdm
from pathlib import Path
from affine import Affine
from shapely import Polygon
from rasterio.crs import CRS
from rasterio.features import sieve, shapes
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url
from skimage.measure import find_contours, approximate_polygon

from geovision.data import Dataset, DatasetConfig
from geovision.data.transforms import SegmentationCompose

import logging
logger = logging.getLogger(__name__)

# NOTE:
# preprocessing:
#   -> saving image tiles to hdf [reduce load time by avoiding RandomCrop during runtime]
#   -> option to save to jpg [reduce load time perhaps, at the cost of quality] 
#   -> Dataset should be able to adapt to these situations dynamically, by checking metadata in the .h5/images for image_shape and image_format

# impl. spatial sampler 
#   -> index_df, spatial_df, tile_size: tuple[int, int], tile_stride: tuple[int, int]

class Inria:
    local = Path.home() / "datasets" / "inria"
    class_names = ("background", "building") 
    loc_names = {
        "supervised": ("vienna", "tyrol-w", "austin", "chicago", "kitsap"),
        "unsupervised":  ("innsbruck", "tyrol-e", "bloomington", "sfo", "bellingham")
    } 
    identity_matrix = np.eye(2, dtype = np.float32)
    default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="stratified",
        tabular_sampler_params={"split_on": "location", "val_frac": 0.15, "test_frac": 0.15},
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        target_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]),
        train_aug=T.Compose([T.RandomCrop(256, pad_if_needed=True), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
        eval_aug=T.RandomResizedCrop(256)
    ) 

    @classmethod
    def extract(cls, src: Literal["inria.fr", "huggingface", "kaggle"]):
        """downloads from :src to :local/staging"""

        valid_sources = ("inria.fr", "huggingface", "kaggle") 
        assert src in valid_sources, f"value error, expected :src to be one of {valid_sources}, got {src}"

        staging_path = fs.get_new_dir(cls.local/"staging") 

        if src == "inria.fr":
            # download multi-part .7z archive from inria.fr to :local/staging
            for url in tqdm([f"https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.00{x}" for x in range(1, 6)]):
                download_url(url, staging_path)
            
            # from aerialimagelabeling.7z.00x extract NEW2-AerialImageDataset.zip (weird way to package .zip inside a multi-part .7z)
            with multivolumefile.open(staging_path/"aerialimagelabeling.7z", mode = "rb") as multi_archive:
                with py7zr.SevenZipFile(multi_archive, mode = "r") as archive:
                    archive.extractall(staging_path)

            # cleanup downloaded aerialimagelabelling.7z.00x files
            for x in range (1, 6):
                (staging_path / f"aerialimagelabeling.7z.00{x}").unlink(missing_ok=True)

        elif src == "huggingface":
            # download imagefolder from huggingface to :local/staging
            raise NotImplementedError 

        elif src == "kaggle":
            # download imagefolder from kaggle to :local/staging
            raise NotImplementedError 

    @classmethod
    def download(cls):
        """
            downloads s3://inria/hdf5/inria.h5 to :local/hdf5. 
            raises FileNotFoundError if s5cmd not installed in environment or file not found on s3 at remote_url
        """
        if subprocess.call(["s5cmd", "cp", remote := "s3://inria/hdf5/inria.h5", local := str(fs.get_new_dir(cls.local/"hdf5"))]) == 0:
            logger.info(f"downloaded {remote} to {local}")
        else:
            raise FileNotFoundError(f"couldn't find {remote}")

    @classmethod
    def load(
            cls, 
            table: Literal["index", "spatial"],
            src: Literal["staging/archive", "staging/imagefolder", "imagefolder", "hdf5"],
            subset: Literal["inria"] = "inria"
        ) -> pd.DataFrame:

        assert src in ("staging/archive", "staging/imagefolder", "imagefolder", "hdf5")
        assert subset == "inria" 

        def _load_metadata(metadata_path: Path, image_paths: list[str | Path]) -> pd.DataFrame:
            metadata = {k: list() for k in ("image_path", "x_off", "y_off", "x_res", "y_res", "crs")}
            for image_path in tqdm(image_paths):
                metadata["image_path"].append(image_path)
                with rio.open(image_path) as raster:
                    metadata["x_off"].append(raster.transform.xoff)
                    metadata["y_off"].append(raster.transform.yoff)
                    metadata["x_res"].append(raster.transform.a)
                    metadata["y_res"].append(raster.transform.e)
                    metadata["crs"].append(raster.crs.to_epsg())

            df = (
                pd.DataFrame(metadata)
                .sort_values("image_path")
                .reset_index(drop = True)
                .assign(image_height = 5000)
                .assign(image_width = 5000)
                .assign(location = lambda df: df["image_path"].apply(lambda x: ''.join([i for i in Path(x).stem if not i.isdigit()])))
                .assign(mask_path = lambda df: df.apply(lambda x: x["image_path"].replace("images", "gt") if x["location"] in cls.loc_names["supervised"] else None, axis = "columns"))
            )
            index_df = df[["image_path", "mask_path", "location"]]
            spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]

            index_df.to_hdf(metadata_path, key = "index", mode = 'a', index = True)
            spatial_df.to_hdf(metadata_path, key = "spatial", mode = 'r+', index = True)
            return index_df if table == "index" else spatial_df

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/"inria.h5", key = table, mode = 'r')
        
        elif src == "staging/imagefolder" or src == "imagefolder":
            imagefolder_path = cls.local / src 
            try:
                return pd.read_hdf(imagefolder_path/"metadata.h5", key = table, mode = 'r')
            except OSError:
                assert table in ("index", "spatial"), \
                    f"not implemented error, expected :table to be one of index or spatial, since this function cannot create {table} metadata"
                image_paths = [x for x in fs.get_valid_dir_err(imagefolder_path).rglob("*.tif") if x.parent != "masks"]
                return _load_metadata(imagefolder_path/"metadata.h5", image_paths)

        elif src == "staging/archive":
            archive_path = fs.get_valid_file_err(cls.local, "staging", "NEW2-AerialImageDataset.zip")
            metadata_path = cls.local / "staging" / "metadata.h5"
            try:
                #raise OSError
                return pd.read_hdf(metadata_path, key = table, mode = 'r')
            except OSError:
                assert table in ("index", "spatial"), \
                    f"not implemented error, expected :table to be one of index or spatial, since this function cannot create {table} metadata"
                with zipfile.ZipFile(archive_path, mode = 'r') as zf:
                    image_paths = [f"zip+file://{str(archive_path)}!{x}" for x in zf.namelist() if x.endswith(".tif") and x.split('/')[-2] != "gt"]
                return _load_metadata(metadata_path, image_paths)

    @classmethod
    def transform(
            cls, 
            src: Literal["staging/archive", "staging/imagefolder"],
            to: Literal["imagefolder", "hdf5"], 
            tile_size: Optional[int | tuple[int, int]] = None,
            tile_stride: Optional[int | tuple[int, int]] = None,
        ):

        assert src in ("staging/archive", "staging/imagefolder")
        assert to in ("imagefolder", "hdf5")

        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            if tile_stride is None:
                tile_stride == (tile_size[0]//2, tile_size[1]//2)
            elif tile_stride is not None:
                if isinstance(tile_stride, int):
                    tile_stride = (tile_stride, tile_stride)

            assert isinstance(tile_size, Sequence) and len(tile_size) == 2
            assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

        def _tile(index_df: pd.DataFrame, spatial_df: pd.DataFrame) -> pd.DataFrame:
            return (
                DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
                .merge(spatial_df, how = 'left', left_index=True, right_index=True)
                .assign(x_off = lambda df: df.apply(lambda col: col["x_off"] + col["x_min"] * col["x_res"], axis = 1))
                .assign(y_off = lambda df: df.apply(lambda col: col["y_off"] + col["y_min"] * col["y_res"], axis = 1))
                .assign(image_height = tile_size[0])
                .assign(image_width = tile_size[1])
                .reset_index(drop = True)
            )
        
        if to == "imagefolder":
            imagefolder_path = fs.get_new_dir(cls.local, "imagefolder")
            sup_images = fs.get_new_dir(imagefolder_path, "sup", "images")
            sup_masks = fs.get_new_dir(imagefolder_path, "sup", "masks")
            unsup_images = fs.get_new_dir(imagefolder_path, "unsup", "images")

            if src == "staging/imagefolder":
                staging_path = fs.get_valid_dir_err(cls.local, "staging")
                if tile_size is None:
                    pass # move images from :local/staging to :local/imagefolder with appropriate directory names
                else:
                    pass # load, tile and paste 
            elif src == "staging/archive":
                archive_path = fs.get_valid_file_err(cls.local, "staging", "NEW2-AerialImageDataset.zip")

                if tile_size is None:
                    with zipfile.ZipFile(archive_path, mode = 'r') as archive:
                        archive.extractall(imagefolder_path)
                    shutil.move(imagefolder_path/"AerialImageDataset"/"test"/"images", unsup_images)
                    shutil.move(imagefolder_path/"AerialImageDataset"/"train"/"images", sup_images)
                    shutil.move(imagefolder_path/"AerialImageDataset"/"train"/"gt", sup_masks)
                    shutil.rmtree(imagefolder_path/"AerialImageDataset")

                    cls.load('index', 'archive', 'inria') \
                    .assign(image_path = lambda df: df["image_path"].apply(lambda x: '/'.join(x.split('/')[-3:]).replace("train", "sup").replace("test", "unsup"))) \
                    .assign(mask_path = lambda df: df["image_path"].apply(lambda x: x.replace("gt", "masks") if x is not None else None)) \
                    .to_hdf(imagefolder_path/"metadata.h5", key = "index", mode = "w")

                    cls.load('spatial', 'archive', 'inria').to_hdf(imagefolder_path/"metadata.h5", key = "spatial", mode = "r+")
                
                else:
                    def _get_tile_path(prefix: Path, row: pd.DataFrame):
                        grandparent_dir, parent_dir, filename = row["image_path"].split('/')[-3:]
                        grandparent_dir = "unsup" if grandparent_dir == "test" else "sup"
                        filename = f"{filename.removesuffix(".tif")}_{row["y_min"]}_{row["x_min"]}.tif"
                        return prefix/grandparent_dir/parent_dir/filename

                    index_df = cls.load('index', 'archive', 'inria')
                    spatial_df = cls.load('spatial', 'archive', 'inria')
                    tiled_df = _tile(index_df, spatial_df)

                    # for each scene (5000x5000x3)
                    for scene in tqdm(tiled_df["image_path"].unique()):

                        # filter all tile rows from that scene  
                        df = tiled_df[tiled_df["image_path"] == scene]

                        # load the scene raster
                        with rio.open(df["image_path"].iloc[0]) as raster:
                            image = raster.read().transpose(1,2,0)
                        
                        # if the image has a corresponding mask, load that as well
                        supervised = False
                        if df["mask_path"].iloc[0] is not None:
                            supervised = True
                            with rio.open(df["mask_path"].iloc[0]) as raster:
                                mask = raster.read().squeeze()

                        kwargs = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'height': tile_size[0], 'width': tile_size[1]}

                        # for each tile in scene 
                        for i, row in df.iterrows():

                            tile_kwargs = kwargs | {
                                "fp": _get_tile_path(imagefolder_path, row), "mode": "w", "count": 3,
                                "crs": CRS.from_epsg(row["crs"]), "transform": Affine(row["x_res"], 0, row["x_off"], 0, row["y_res"], row["y_off"])
                            }

                            with rio.open(**tile_kwargs) as raster:
                                raster.write(cls.read_tile(image, row).transpose(2,0,1))

                            if supervised:
                                tile_kwargs["fp"] = Path(str(tile_kwargs["fp"]).replace("images", "masks"))
                                tile_kwargs["count"] = 1
                                with rio.open(**tile_kwargs) as raster:
                                    raster.write(cls.read_tile(mask, row)[np.newaxis,:,:])

                    tiled_df["image_path"] = tiled_df.apply(lambda row: _get_tile_path(imagefolder_path, row), axis = "columns")
                    tiled_df["mask_path"] = tiled_df["image_path"].apply(lambda x: x.replace("gt", "masks") if x is not None else None)

                    tiled_df[["image_path", "mask_path", "location"]].to_hdf(imagefolder_path/"metadata.h5", key = "index", mode = "w")
                    tiled_df.drop(columns=["image_path", "mask_path", "location"]).to_hdf(imagefolder_path/"metadata.h5", key = "spatial", mode = "r+")
                    
        elif to == "hdf5":
            hdf5_path = fs.get_new_dir(cls.local, "hdf5") / "inria.h5"

            if src == "staging/archive":
                archive_path = fs.get_valid_file_err(cls.local, "staging", "NEW2-AerialImageDataset.zip")
                index_df = cls.load("index", "archive", 'inria')
                spatial_df = cls.load("spatial", "archive", 'inria')

                if tile_size is None:
                    pass
                else:
                    pass

            elif src == "staging/imagefolder":
                imagefolder_path = fs.get_valid_dir_err(cls.local, "staging")
                index_df = cls.load("index", "staging/imagefolder", 'inria')
                spatial_df = cls.load("spatial", "staging/imagefolder", 'inria')

                if tile_size is None:
                    sup_index_df = index_df[index_df["mask_path"].notna()].reset_index(drop = True)
                    sup_spatial_df = spatial_df[index_df["mask_path"].notna()].reset_index(drop = True)

                    unsup_index_df = index_df[index_df["mask_path"].isna()].drop(columns="mask_path").reset_index(drop=True)
                    unsup_spatial_df = spatial_df[index_df["mask_path"].isna()].reset_index(drop = True)

                    with h5py.File(hdf5_path, mode = 'w') as file:
                        sup_images = file.create_dataset("sup/images", (180, 5000, 5000, 3), dtype = np.uint8)
                        sup_masks = file.create_dataset("sup/masks", 180, dtype = h5py.special_dtype(vlen=np.uint8))

                        sup_images.attrs["image_size"] = (5000,5000)
                        for idx, row in tqdm(sup_index_df.iterrows(), total = 180):
                            sup_images[idx] = iio.imread(imagefolder_path / row["image_path"], extension=".tif")
                            sup_masks[idx] = cls.encode_binary_mask(iio.imread(imagefolder_path / row["mask_path"], extension=".tif"))
                        
                        unsup_images = file.create_dataset("unsup/images", (180, 5000, 5000, 3), dtype = np.uint8)
                        unsup_images.attrs["image_size"] = (5000,5000)
                        for idx, row in tqdm(unsup_index_df.iterrows(), total = 180):
                            unsup_images[idx] = iio.imread(imagefolder_path / row["image_path"], extension=".tif")

                    sup_index_df.drop(columns = "mask_path").to_hdf(hdf5_path, key = "sup/index", mode = 'r+')
                    sup_spatial_df.to_hdf(hdf5_path, key = "sup/spatial", mode = 'r+')

                    unsup_index_df.to_hdf(hdf5_path, key = "sup/index", mode = 'r+')
                    unsup_spatial_df.to_hdf(hdf5_path, key = "unsup/spatial", mode = 'r+')

                else:
                    tiled_df = _tile(index_df, spatial_df)

                    sup_tiled_df = tiled_df[tiled_df["mask_path"].notna()].reset_index(drop=True)
                    unsup_tiled_df = tiled_df[tiled_df["mask_path"].isna()].drop(columns="mask_path").reset_index(drop=True)

                    with h5py.File(hdf5_path, mode = 'w') as file:
                        sup_images = file.create_dataset("sup/images", (len(sup_tiled_df), *tile_size, 3), dtype = np.uint8)
                        sup_masks = file.create_dataset("sup/masks", len(sup_tiled_df), dtype=h5py.special_dtype(vlen=np.uint8))
                        sup_images.attrs["image_size"] = tile_size

                        unsup_images = file.create_dataset("unsup/images", (len(unsup_tiled_df), *tile_size, 3), dtype = np.uint8)
                        unsup_images.attrs["image_size"] = tile_size

                        for scene in tqdm(sup_tiled_df["image_path"].unique()):
                            df = sup_tiled_df[sup_tiled_df["index"] == scene]
                            with rio.open(df["image_path"].iloc[0]) as raster:
                                image = raster.read().transpose(1,2,0)
                            with rio.open(df["mask_path"].iloc[0]) as raster:
                                mask = raster.read().squeeze()
                            for i, row in df.iterrows():
                                sup_images[i] = cls.read_tile(image, row)
                                sup_masks[i] = cls.encode_binary_mask(cls.read_tile(mask, row))

                        for scene in tqdm(unsup_tiled_df["image_path"].unique()):
                            df = unsup_tiled_df[unsup_tiled_df["index"] == scene]
                            with rio.open(df["image_path"].iloc[0]) as raster:
                                image = raster.read().transpose(1,2,0)
                            for i, row in df.iterrows():
                                unsup_images[i] = cls.read_tile(image, row)

                    sup_images[["image_path", "location"]] \
                    .assign(image_path = lambda df: df.apply(lambda x: f"{x["image_path"].split('/')[-1].removesuffix('.tif')}_{x["y_min"]}_{x["x_min"]}"), axis = 1) \
                    .to_hdf(hdf5_path, key = "sup/index", mode = 'r+')

                    sup_images \
                    .drop(columns = ["image_path", "mask_path", "location"]) \
                    .to_hdf(hdf5_path, key = "sup/spatial", mode = 'r+')

                    sup_images[["image_path", "location"]] \
                    .assign(image_path = lambda df: df.apply(lambda x: f"{x["image_path"].split('/')[-1].removesuffix('.tif')}_{x["y_min"]}_{x["x_min"]}"), axis = 1) \
                    .to_hdf(hdf5_path, key = "unsup/index", mode = 'r+')

                    sup_images \
                    .drop(columns = ["image_path", "location"]) \
                    .to_hdf(hdf5_path, key = "unsup/spatial", mode = 'r+')
                
    @staticmethod
    def encode_binary_mask(mask : NDArray) -> NDArray:
        # NOTE: might have to change 255 to mask.max() to adapt to more bands
        mask = np.where(mask.flatten() == 255, 1, 0)
        indices = np.nonzero(np.diff(mask, n = 1))[0] + 1
        indices = np.concat(([0], indices, [len(mask)]))
        rle = np.diff(indices, n = 1)
        if mask[0] == 1:
            return np.concat(([0], rle))
        return rle

    @staticmethod
    def decode_binary_mask(rle: NDArray, shape: tuple):
        mask = np.zeros(rle.sum(), dtype = np.uint8)
        idx, fill = 0, 0
        for run in rle:
            new_idx = idx + run
            if fill == 1:
                mask[idx: new_idx] = 1 
                idx, fill = new_idx, 0
            else:
                idx, fill = new_idx, 1
        return mask.reshape(shape)

    @staticmethod
    def read_tile(scene: NDArray, row: pd.Series) -> NDArray:
        def _pad(n: int) -> tuple[int, int]:
            return (n//2, n//2) if n % 2 == 0 else (n//2, (n//2) + 1)

        tile = scene[row["y_min"] : min(row["y_max"], 5000), row["x_min"] : min(row["x_max"], 5000)]
        tile_size = (int(row["y_max"] - row["y_min"]), int(row["x_max"] - row["x_min"]), 3)
        if scene.ndim == 2:
            tile_size = tile_size[:2] 

        if tile.shape != tile_size: 
            pad_width = [_pad(tile_size[0] - tile.shape[0]), _pad(tile_size[1] - tile.shape[1]), (0, 0)]
            tile = np.pad(array = tile, pad_width = pad_width[:2] if tile.ndim == 2 else pad_width)
        return tile
    
InriaIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "mask_path": pa.Column(str, coerce=True),
        "location": pa.Column(str, coerce=True),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    },
    index=pa.Index(int)
)

class Inria_Building_Segmentation_Imagefolder(Dataset):
    name = "inria"
    task = "segmentation"
    subtask = "semantic"
    storage = "imagefolder"
    class_names = ("background", "building") 
    num_classes = 2 
    root = Inria.local/"imagefolder"
    schema = InriaIndexSchema 
    config = Inria.default_config
    loader = Inria.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=True)
        self.identity_matrix = np.eye(self.num_classes, dtype = np.uint8)

        self.crop = False 
        if self.config.spatial_sampler_name is not None: 
            self.crop = True
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        image = iio.imread(idx_row["image_path"], extension=".tif")
        mask = iio.imread(idx_row["mask_path"], extension=".tif").squeeze()
        if self.crop:
            image, mask = Inria.read_tile(image, idx_row), Inria.read_tile(mask, idx_row)
        mask = self.identity_matrix[np.clip(mask, 0, 1)] # mask.shape = (H, W, num_channels)
        image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        #print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        #print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())
        return image, mask, idx_row["df_idx"]

class Inria_Building_Segmentation_HDF5(Dataset):
    name = "inria"
    task = "segmentation"
    subtask = "semantic"
    storage = "hdf5"
    class_names = ("background", "building") 
    num_classes = 2 
    root = Inria.local/"hdf5"/"inria_supervised.h5"
    schema = InriaIndexSchema 
    config = Inria.default_config
    loader = Inria.load
    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=False)
        self.identity_matrix = np.eye(self.num_classes, dtype = np.uint8)

        with h5py.File(self.root) as f:
            self.mask_shape = f["images"].attrs.get("image_size", (5000, 5000))

        self.crop = False 
        if self.config.spatial_sampler_name is not None: 
            self.crop = True
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode = "r") as f:
            image = f["images"][idx_row["df_idx"]]
            mask = Inria.decode_binary_mask(f["masks"][idx_row["df_idx"]], shape = self.mask_shape) # mask: (H, W)
        if self.crop:
            image, mask = Inria.read_tile(image, idx_row), Inria.read_tile(mask, idx_row)
        mask = self.identity_matrix[np.clip(mask, 0, 1)] # mask: (H, W, num_channels)
        image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        #print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        #print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())
        return image, mask, idx_row["df_idx"]