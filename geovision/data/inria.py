from typing import Literal, Optional, Callable, Sequence
from numpy.typing import NDArray

import h5py
import py7zr
import torch
import shutil
import zipfile
import subprocess
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
    identity_matrix = np.eye(2, dtype = np.float32)
    default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="stratified",
        tabular_sampler_params={"split_on": "location", "val_frac": 0.15, "test_frac": 0.15},
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        target_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]),
        train_aug=SegmentationCompose([T.RandomCrop(256, pad_if_needed=True), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
    ) 

    @classmethod
    def extract(cls, src: Literal["inria.fr", "huggingface", "kaggle"]):
        """downloads from :src to :local/imagefolder"""
        import multivolumefile

        valid_sources = ("inria.fr", "huggingface", "kaggle") 
        assert src in valid_sources, f"value error, expected :src to be one of {valid_sources}, got {src}"

        staging_path = fs.get_new_dir(cls.local, "staging") 
        imagefolder_path = fs.get_new_dir(cls.local, "imagefolder")
        if src == "inria.fr":
            urls = (
                "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
                "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
                "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
                "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
                "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005"
            )
            # download urls
            for url in tqdm(urls):
                download_url(url, staging_path)
            
            # extract NEW2-AerialImageDataset.zip from aerialimagelabeling.7z (weird way to package)
            with multivolumefile.open(staging_path / "aerialimagelabeling.7z", mode = "rb") as multi_archive:
                with py7zr.SevenZipFile(multi_archive, mode = "r") as archive:
                    archive.extractall(staging_path)
                # remove aerialimagelabelling.7z.00x
                for url in urls:
                    (staging_path / url.split('/')[-1]).unlink(missing_ok=True)

        else:
            raise NotImplementedError 

    @classmethod
    def download(cls, subset: Literal["inria"], supervised: bool = True):
        """downloads inria_:subset.h5 from s3://inria/hdf5/ to :local/hdf5. raises FileNotFoundError if s5cmd not installed in environment or file not in :remote"""
        assert subset in ("inria",), f"value error, expected :subset to be one of supervised or unsupervised, got {subset}"
        remote = f"s3://inria/hdf5/{subset}_{"supervised" if supervised else "unsupervised"}.h5"
        ret = subprocess.call(["s5cmd", "cp", remote, str(fs.get_new_dir(cls.local, "hdf5"))])
        if ret != 0:
            raise FileNotFoundError(f"couldn't find {remote}")

    @classmethod
    def load(cls, table: Literal["index", "spatial"], src: Literal["archive", "imagefolder", "hdf5"], subset: Literal["inria"], supervised: bool = True) -> pd.DataFrame:

        assert src in ("archive", "imagefolder", "hdf5")
        assert subset == "inria" 

        def _load_metadata(metadata_path: Path, image_paths: list[str | Path]) -> pd.DataFrame:
            metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}
            for image_path in image_paths:
                metadata["image_path"].append(image_path)
                with rio.open(image_path) as raster:
                    metadata["image_width"].append(raster.width)
                    metadata["image_height"].append(raster.height)
                    metadata["x_off"].append(raster.transform.xoff)
                    metadata["y_off"].append(raster.transform.yoff)
                    metadata["x_res"].append(raster.transform.a)
                    metadata["y_res"].append(raster.transform.e)
                    metadata["crs"].append(raster.crs.to_epsg())

            df = pd.DataFrame(metadata).sort_values("image_path").reset_index(drop = True)
            df["location"] = df["image_path"].apply(lambda x: ''.join([i for i in Path(x).stem if not i.isdigit()]))

            if supervised:
                df["mask_path"] = df["image_path"].apply(lambda x: x.replace("images", "gt"))
                index_df = df[["image_path", "mask_path", "location"]]
            else:
                df["image_path"] = df["image_path"].apply(lambda x: x.split('/')[-1])
                index_df = df[["image_path", "location"]]

            spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]

            index_df.to_hdf(metadata_path, key = "index", mode = 'a', index = True)
            spatial_df.to_hdf(metadata_path, key = "spatial", mode = 'r+', index = True)
            return index_df if table == "index" else spatial_df

        if src == "archive":
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
                if supervised:
                    image_paths = [x for x in image_paths if x.split('/')[-3] != "test"]
                else:
                    image_paths = [x for x in image_paths if x.split('/')[-3] == "test"]
                return _load_metadata(metadata_path, image_paths)

        elif src == "imagefolder":
            metadata_path = cls.local / src / "metadata.h5" 
            try:
                return pd.read_hdf(metadata_path, key = table, mode = 'r')
            except OSError:
                assert table in ("index", "spatial"), \
                    f"not implemented error, expected :table to be one of index or spatial, since this function cannot create {table} metadata"
                image_paths = (fs.get_valid_dir_err(metadata_path.parent) / ("images" if supervised else "unsup")).rglob("*.tif")
                return _load_metadata(metadata_path, image_paths)

        elif src == "hdf5":
            return pd.read_hdf(cls.local / "hdf5" / f"inria_{"supervised" if supervised else "unsupervised"}.h5", key = table, mode = 'r')

    @classmethod
    def transform(
            cls, 
            to: Literal["imagefolder", "hdf5"], 
            subset: Literal["inria"], 
            tile_size: Optional[int | tuple[int, int]] = None,
            tile_stride: Optional[int | tuple[int, int]] = None,
            supervised: bool = True
        ):
        """transforms the :subset of dataset to :to and applies :transforms to each (image, mask) if provided"""
        assert to in ("imagefolder", "hdf5")
        assert subset == "inria"
        archive_path = fs.get_valid_file_err(cls.local, "staging", "NEW2-AerialImageDataset.zip")

        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            assert isinstance(tile_size, Sequence) and len(tile_size) == 2

        elif tile_stride is not None:
            if isinstance(tile_stride, int):
                tile_stride = (tile_stride, tile_stride)
            assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

        def _tile(index_df: pd.DataFrame, spatial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            index_columns = index_df.columns
            index_df = (
                DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
                .merge(spatial_df, how = 'left', left_index=True, right_index=True)
                .assign(x_off = lambda df: df.apply(lambda col: col["x_off"] + col["tile_tl_0"] * col["x_res"], axis = 1))
                .assign(y_off = lambda df: df.apply(lambda col: col["y_off"] + col["tile_tl_0"] * col["y_res"], axis = 1))
                .assign(image_height = tile_size[0])
                .assign(image_width = tile_size[1])
            )
            return index_df[index_columns], index_df[index_df.columns.difference(index_columns)]


        if to == "imagefolder":
            imagefolder_path = fs.get_new_dir(cls.local, "imagefolder")
            if supervised:
                fs.get_new_dir(imagefolder_path, "train", "images")
                fs.get_new_dir(imagefolder_path, "train", "masks")
            else:
                fs.get_new_dir(imagefolder_path, "unsup", "images")

            if tile_stride is not None and tile_size is not None:
                index_df = cls.load('index', 'archive', subset, supervised=supervised)
                spatial_df = cls.load('spatial', 'archive', subset, supervised=supervised)
                tiled_df = pd.concat(_tile(index_df, spatial_df), axis = 1).reset_index(drop = False)

                # For Each (Original Scene) Image
                for idx in tqdm(tiled_df["index"].unique()):
                    # Filter and Load Image 
                    df = tiled_df[tiled_df["index"] == idx]
                    with rio.open(df["image_path"].iloc[0]) as raster:
                        image = raster.read().transpose(1,2,0)
                    if supervised:
                        with rio.open(df["mask_path"].iloc[0]) as raster:
                            mask = raster.read().squeeze()
                    kwargs = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'height': tile_size[0], 'width': tile_size[1]}

                    # For Each Tile in Scene
                    for i, row in df.iterrows():

                        grandparent_dir, parent_dir, filename = row["image_path"].split('/')[-3:]
                        filename = f"{filename.removesuffix(".tif")}_{row["tile_tl_0"]}_{row["tile_tl_1"]}.tif"
                        if not supervised:
                            grandparent_dir = "unsup"

                        tile_kwargs = kwargs | {
                            "fp": imagefolder_path / '/'.join([grandparent_dir, parent_dir, filename]), 
                            "mode": "w", 
                            "count": 3,
                            "crs": CRS.from_epsg(row["crs"]), "transform": Affine(row["x_res"], 0, row["x_off"], 0, row["y_res"], row["y_off"])
                        }

                        with rio.open(**tile_kwargs) as raster:
                            raster.write(cls.read_tile(image, row).transpose(2,0,1))

                        if supervised:
                            tile_kwargs["fp"] = Path(str(tile_kwargs["fp"]).replace("images", "masks"))
                            tile_kwargs["count"] = 1
                            with rio.open(**tile_kwargs) as raster:
                                raster.write(cls.read_tile(mask, row)[np.newaxis,:,:])
            else:
                with zipfile.ZipFile(archive_path, mode = 'r') as archive:
                    archive.extractall(imagefolder_path)
                shutil.move(imagefolder_path / "AerialImageDataset" / "test" / "images", imagefolder_path / "unsup")
                shutil.move(imagefolder_path / "AerialImageDataset" / "train" / "images", imagefolder_path / "images")
                shutil.move(imagefolder_path / "AerialImageDataset" / "train" / "gt", imagefolder_path / "masks")
                shutil.rmtree(imagefolder_path / "AerialImageDataset")

        elif to == "hdf5":
            imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder")
            hdf5_path = fs.get_new_dir(cls.local, "hdf5") / f"inria_{"supervised" if supervised else "unsupervised"}.h5"
            index_df = cls.load("index", "archive", subset, supervised)
            spatial_df = cls.load("spatial", "archive", subset, supervised)

            if tile_size is not None and tile_stride is not None:
                tiled_df = pd.concat(_tile(index_df, spatial_df), axis = 1).reset_index(drop = False)
                with h5py.File(hdf5_path, mode = 'w') as file:
                    images = file.create_dataset("images", (len(tiled_df), *tile_size, 3), dtype = np.uint8)
                    images.attrs["image_size"] = tile_size
                    if supervised:
                        masks = file.create_dataset("masks", len(tiled_df), dtype=h5py.special_dtype(vlen=np.uint8))
                    for idx in tqdm(tiled_df["index"].unique()):
                        df = tiled_df[tiled_df["index"] == idx]
                        with rio.open(df["image_path"].iloc[0]) as raster:
                            image = raster.read().transpose(1,2,0)
                        if supervised:
                            with rio.open(df["mask_path"].iloc[0]) as raster:
                                mask = raster.read().squeeze()
                        for i, row in df.iterrows():
                            images[i] = cls.read_tile(image, row)
                            if supervised:
                                masks[i] = cls.encode_binary_mask(cls.read_tile(mask, row))
                if supervised:
                    tiled_df = tiled_df.drop(columns = "mask_path")
                tiled_df["image_path"] = tiled_df.apply(lambda col: f"{col["image_path"].split('/')[-1].removesuffix('.tif')}_{col["tile_tl_0"]}_{col["tile_tl_1"]}.tif", axis = 1)
                tiled_df[["image_path", "location"]].to_hdf(hdf5_path, key = 'index', mode = 'r+')
                tiled_df.drop(columns=["image_path", "location", "index"]).to_hdf(hdf5_path, key = 'spatial', mode = 'r+')

            else:
                with h5py.File(hdf5_path, mode = 'w') as file:
                    images = file.create_dataset("images", (180, 5000, 5000, 3), dtype = np.uint8)
                    images.attrs["image_size"] = (5000,5000)
                    if supervised:
                        masks = file.create_dataset("masks", 180, dtype = h5py.special_dtype(vlen=np.uint8))
                    for idx, row in tqdm(index_df.iterrows(), total = 180):
                        images[idx] = iio.imread(imagefolder_path / row["image_path"], extension=".tif")
                        if supervised:
                            masks[idx] = cls.encode_binary_mask(iio.imread(imagefolder_path / row["mask_path"], extension=".tif"))
                if supervised:
                    index_df = index_df.drop(columns = "mask_path")
                index_df["image_path"] = index_df["image_path"].apply(lambda x: '/'.join(x.split('/')[-3]))
                index_df.to_hdf(hdf5_path, key = "index", mode = 'r+')
                spatial_df.to_hdf(hdf5_path, key = "spatial", mode = 'r+')

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

        tile = scene[row["tile_tl_0"] : min(row["tile_br_0"], 5000), row["tile_tl_1"] : min(row["tile_br_1"], 5000)]
        tile_size = (row["tile_br_0"] - row["tile_tl_0"], row["tile_br_1"] - row["tile_tl_1"], 3)
        if scene.ndim == 2:
            tile_size = tile_size[:2] 

        if tile.shape != tile_size: 
            pad_width = [_pad(tile_size[0] - tile.shape[0]), _pad(tile_size[1] - tile.shape[1]), (0, 0)]
            tile = np.pad(array = tile, pad_width = (pad_width[:2] if tile.ndim == 2 else pad_width))
        return tile
    
    @staticmethod
    def polygonize(
            mask: NDArray,
            transform: Optional[Affine] = None,
            method: Literal["manual", "gdal"] = "manual",
            min_pixels: int = 100,
            connectivity: Literal[4, 8] = 4,
            approximation_tolerance: Optional[float] = None
    ) -> list[Polygon]:

        if method == "manual":
            mask = sieve(mask, size=min_pixels, connectivity=connectivity)

            polygons = list()
            for contour in find_contours(mask, fully_connected="low" if connectivity == 4 else "high"):
                contour = approximate_polygon(contour, tolerance=approximation_tolerance)
                if transform is not None:
                    contour = [transform*(vertex[1], vertex[0]) for vertex in contour]
                    # vertices = np.matrix(vertices) # vertices = [[y1, x1], [y2, x2], ..., [yn, xn]], shape = (#vertices, 2) [y,x]
                    # vertices[:, [0, 1]] = vertices[:, [1, 0]] # vertices = [[x1, y1], [x2, y2], ..., [xn, yn]], shape = (#vertices, 2) [x,y]
                    # vertices = np.c_[vertices, np.ones(vertices.shape[0])] # vertices = [[x1, y1, 1], [x2, y2, 1], ..., [xn, yn, 1]], shape = (#vertices, 3) [x,y,1]
                    # vertices = np.transpose(vertices) # shape = (3, #vertices) [each vertex is now a column vector]
                    # vertices = np.matmul(transform, vertices) # shape = (3, #vertices) []
                    # vertices = np.transpose(vertices[:2]) # shape = (#vertices, 2)
                polygon = Polygon(contour)
                if polygon.is_valid:
                    polygons.append(polygon)
            return polygons

        elif method == "gdal":
            # https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.shapes

            mask = sieve(mask, size=min_pixels, connectivity=connectivity)
            polygons = shapes(mask, connectivity=connectivity, transform=transform)
            return list(polygons)

    @staticmethod
    def bounds(image: NDArray, row: pd.Series) -> Polygon: ...

InriaIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "mask_path": pa.Column(str, coerce=True),
        "location": pa.Column(str, coerce=True),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    },
    index=pa.Index(int, unique=True)
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