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

import logging
logger = logging.getLogger(__name__)

# SPECTRAL_IMAGE: L2A (Orthorectified and Atmospherically Corrected Reflectance Values, 128x128x224 little-endian int16) 
# METADATA
# STAC_JSON

# QL_PIXELMASK: defective pixels mask
# QL_SWIR: 
# QL_VNIR:
# QL_VNIR_THUMBNAIL: 
# QL_QUALITY_CIRRUS:
# QL_QUALITY_CLASSES:
# QL_QUALITY_SNOW:
# QL_QUALITY_HAZE:
# QL_QUALITY_CLOUDSHADOW:
# QL_QUALITY_CLOUD:
# QL_QUALITY_TESTFLAGS:

class HySpecNet11k:
    local = Path.home() / "datasets" / "hyspecnet11k"
    num_images = 11483

    total_bands = 224
    water_absorption_bands = ((127,142), (161,168)) # 1-indexed, last index not included
    band_idxs = tuple(sorted(set(range(1, 225)) - set(range(127,142)) - set(range(161, 168)))) # subset of bands to read using rasterio
    num_bands = 202 # after removing water absorption bands

    index_df_schema = pa.DataFrameSchema(
        columns={
            "image_path": pa.Column(str, coerce=True),
            "file_name": pa.Column(str, coerce=True),
        },
        index=pa.Index(int)
    )

    default_config = DatasetConfig(
        random_seed=42,
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        train_aug=T.Compose([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
        eval_aug=T.Identity()
    )

    @classmethod
    def extract(cls): ...

    @classmethod
    def download(cls): ...

    @classmethod
    def load(cls, table: Literal["index", "spatial"], src: Literal["staging", "imagefolder", "hdf5"], subset: Literal["hyspecnet11k"] = "hyspecnet11k") -> pd.DataFrame:
        assert src in ("staging", "imagefolder", "hdf5")
        assert subset == "hyspecnet11k"

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/"hyspecnet11k.h5", key = table, mode = 'r')
        
        elif src == "imagefolder":
            return pd.read_hdf(fs.get_valid_dir_err(cls.local, "imagefolder")/"metadata.h5", key = table, mode = 'r')

        elif src == "staging":
            staging_path = fs.get_valid_dir_err(cls.local, "staging")
            try:
                return pd.read_hdf(staging_path/"metadata.h5", key = table, mode = 'r')

            except OSError:
                assert table in ("index", "spatial"), \
                        f"not implemented error, expected :table to be one of index or spatial, since this function cannot create {table} metadata"

                def _get_image_name_col(path_column: pd.Series) -> pd.Series:
                    return path_column.str.split('/').str[-1].str.removeprefix("ENMAP01-____L2A-")
                
                def _get_splits_df(split: Literal["easy", "hard"]) -> pd.DataFrame:
                    return (
                        pd.concat([pd.read_csv(x, names = ["image_path"]).assign(split = x.stem) for x in (staging_path/"splits"/split).iterdir()])
                        .rename(columns = {"split" : f"{split}_split"})
                        .assign(image_name = lambda df: _get_image_name_col(df["image_path"].astype(str).str.removesuffix("-DATA.npy")))
                        .set_index("image_name")
                        .drop(columns="image_path")
                    )
                
                def _get_product_df(image_names: pd.Series) -> pd.DataFrame: 
                    product_col_names = ("datatake_id", "datatake_start_time", "tile_id", "product_version", "processing_time", "file_name")
                    return (
                        image_names.str.split(r'[-_]', expand = True, n = 5)
                        .rename(columns = {x:y for x,y in zip(range(6), product_col_names)})
                        .assign(datatake_start_time = lambda df: pd.to_datetime(df["datatake_start_time"]))
                        .assign(processing_time = lambda df: pd.to_datetime(df["processing_time"]))
                    )
                
                index_df = (
                    pd.DataFrame(staging_path.rglob("*SPECTRAL_IMAGE.TIF"), columns = ["image_path"])
                    .assign(image_name = lambda df: _get_image_name_col(df["image_path"].astype(str).str.removesuffix("-SPECTRAL_IMAGE.TIF")))
                    .set_index("image_name")
                    .merge(right = _get_splits_df("easy"), how = "left", left_index=True, right_index=True)
                    .merge(right = _get_splits_df("hard"), how = "left", left_index=True, right_index=True)
                )
                index_df = index_df.merge(_get_product_df(index_df.index.to_series()), how = "left", left_index=True, right_index=True)
                index_df = index_df.sort_index(ascending=True).reset_index(drop=False)

                spatial_metadata = {k: list() for k in ("x_off", "y_off", "x_res", "y_res", "crs")}
                for image_path in tqdm(index_df["image_path"]):
                    with rio.open(staging_path / "patches" / image_path) as raster:
                        spatial_metadata["x_off"].append(raster.transform.xoff)
                        spatial_metadata["y_off"].append(raster.transform.yoff)
                        spatial_metadata["x_res"].append(raster.transform.a)
                        spatial_metadata["y_res"].append(raster.transform.e)
                        spatial_metadata["crs"].append(raster.crs.to_epsg())
                spatial_df = pd.DataFrame(spatial_metadata).assign(image_height = 128).assign(image_width = 128)

                index_df.to_hdf(staging_path/"metadata.h5", key = "index", mode = "w")
                spatial_df.to_hdf(staging_path/"metadata.h5", key = "spatial", mode = "r+")
            
    @classmethod
    def transform(cls, to: Literal["imagefolder", "hdf5"], subset: Literal["hyspecnet11k"] = "hyspecnet11k", drop_water_abs_bands: bool = True): 
        assert to in ("imagefolder", "hdf5")
        assert subset == "hyspecnet11k"

        index_df = cls.load('index', 'staging', subset)
        spatial_df = cls.load('spatial', 'staging', subset)

        if to == "imagefolder": # copy to :local/imagefolder/images, and create :local/imagefolder/metadata.h5 with image_paths updated
            images_path = fs.get_new_dir(cls.local / "imagefolder" / "images")

            if drop_water_abs_bands:
                for _, row in tqdm(index_df.iterrows(), total = len(index_df)):
                    with rio.open(row["image_path"]) as raster:
                        with rio.open(images_path/f"{row["image_name"]}.tif", mode = 'w', **(raster.profile | {"count": len(cls.band_idxs)})) as out:
                            out.write(raster.read(cls.band_idxs))
            else:
                shutil.copy(row["image_path"], images_path/f"{row["image_name"]}.tif")

            index_df["image_path"] = index_df["image_name"].astype(str) + ".tif"
            index_df.drop(columns=["image_name"]).to_hdf(images_path.parent/"metadata.h5", key = "index", mode = "w")
            spatial_df.to_hdf(images_path.parent/"metadata.h5", key = "spatial", mode = "r+")
           
        elif to == "hdf5": 
            hdf5_path = fs.get_new_dir(cls.local / "hdf5") / "hyspecnet11k.h5"

            with h5py.File(hdf5_path, mode = 'w') as f:
                bands = cls.num_bands if drop_water_abs_bands else cls.total_bands

                images = f.create_dataset("images", shape = (len(index_df), bands, 128, 128), dtype = np.int16)
                for idx, row in tqdm(index_df.iterrows(), total = len(index_df)):
                    with rio.open(row["image_path"]) as raster:
                        images[idx] = raster.read(cls.band_idxs)
            
            index_df.to_hdf(hdf5_path, key = 'index', mode = 'r+')
            spatial_df.to_hdf(hdf5_path, key = "spatial", mode = 'r+')
    
class HySpecNet11k_Imagefolder(Dataset):
    name = "hyspecnet11k"
    task = "unsupervised"
    subtask = "pretraining"
    storage = "imagefolder"
    class_names = tuple() 
    num_classes = 0 
    root = HySpecNet11k.local/"imagefolder"/"images"
    schema = HySpecNet11k.index_df_schema 
    config = HySpecNet11k.default_config
    loader = HySpecNet11k.load
    metadata_group_prefix = None 

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"], config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.index_df.assign(df_idx = lambda df: df.index).reset_index(drop = True)
        self.df["image_path"] = self.df["image_path"].apply(lambda x: str(Path(self.root, x)))

        # NOTE: Temporary, remove if entire imagefolder is extracted
        image_exists_filter = self.df["image_path"].apply(lambda x: Path(x).is_file())
        self.df = self.df[image_exists_filter].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        image: NDArray = iio.imread(idx_row["image_path"], extension=".tif") # C, H, W

        if self.split != "all":
            image = image.transpose(1,2,0) # H, W, C
            image = self.config.image_pre(image) # C, H, W
        if self.split in ("train", "trainvaltest"):
            image = self.config.train_aug(image)
        elif self.split in ("val", "test"):
            image = self.config.eval_aug(image)

        return image, idx_row["df_idx"]

class HySpecNet11k_HDF5(Dataset):
    name = "hyspecnet11k"
    task = "unsupervised"
    subtask = "pretraining"
    storage = "hdf5"
    class_names = tuple() 
    num_classes = 0 
    root = HySpecNet11k.local/"hdf5"/"hyspecnet11k.h5"
    schema = HySpecNet11k.index_df_schema 
    config = HySpecNet11k.default_config
    loader = HySpecNet11k.load
    metadata_group_prefix = None 

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"], config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.index_df.assign(df_idx = lambda df: df.index).drop(columns = ["image_path"]).reset_index(drop = True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        idx_row = self.df.iloc[idx]

        with h5py.File(self.root, mode = 'r') as f:
            image: NDArray = f["images"][idx_row["df_idx"]] # C, H, W

        if self.split != "all":
            image = image.transpose(1,2,0)
            image = self.config.image_pre(image)
        if self.split in ("train", "trainvaltest"):
            image = self.config.train_aug(image)
        elif self.split in ("val", "test"):
            image = self.config.eval_aug(image)

        return image, idx_row["df_idx"]


