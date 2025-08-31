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
import imageio.v3 as iio  # noqa: F401
import torchvision.transforms.v2 as T

from tqdm import tqdm
from pathlib import Path
from affine import Affine
from shapely import Polygon
from rasterio.crs import CRS
from rasterio.features import sieve, shapes
from multiprocessing import Pool, cpu_count
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url
from skimage.measure import find_contours, approximate_polygon

from geovision.data import Dataset, DatasetConfig
from geovision.data.transforms import SegmentationCompose

import logging
logger = logging.getLogger(__name__)

# image_patches: L2A (orthorectified, atmospherically corrected, absorption bands removed reflectance values) (202x128x128 little-endian int16) 

class SpectralEarth:
    local: Path = Path.home() / "datasets" / "spectral_earth"
    num_bands = 202 # after removing water absorption bands
    
    @classmethod
    def extract(cls): ...

    @classmethod
    def download(cls): ...

    @classmethod
    def load(
            cls, 
            table: Literal["index"],
            src: Literal["archive", "imagefolder", "hdf5"],
            subset: Literal["nlcd", "cdl", "corine", "unsup"]
        ) -> pd.DataFrame:

        assert src in ("staging", "imagefolder", "hdf5")
        assert subset in ("nlcd", "cdl", "corine")

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/"spectral_earth.h5", key = table, mode = 'r')
        
        elif src == "staging":
            return pd.read_hdf(fs.get_valid_dir_err(cls.local, "staging")/"metadata.h5", key = table, mode = 'r')

        elif src == "imagefolder":
            imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder")
            try:
                return pd.read_hdf(imagefolder_path/"metadata.h5", key = subset, mode = 'r')[:1000]

            except OSError:
                assert table in ("index"), \
                    f"not implemented error, expected :table to be index, since this function cannot create {table} metadata"
                    
                df = pd.DataFrame(imagefolder_path.rglob("*.tif"), columns = ["image_path"])
                df["dataset"] = df["image_path"].astype(str).str.split('/').str[-3]
                df["image_name"] = df["image_path"].astype(str).str.split('/').str[-2:].str.join('/').str.removesuffix('.tif')
                df = df.drop(columns = "image_path")

                cdl_df, corine_df, _, nlcd_df = (d.reset_index(drop=True).drop(columns="dataset") for _, d in df.groupby("dataset", sort = True))

                cdl_df.to_hdf(imagefolder_path/"metadata.h5", key = "cdl", mode = "w")
                corine_df.to_hdf(imagefolder_path/"metadata.h5", key = "corine", mode = "r+")
                nlcd_df.to_hdf(imagefolder_path/"metadata.h5", key = "nlcd", mode = "r+")

                if subset == "cdl":
                    return cdl_df
                elif subset == "corine":
                    return corine_df
                elif subset == "nlcd":
                    return nlcd_df
                
    @classmethod
    def transform(cls, to: Literal["imagefolder", "hdf5"], subset: Literal["sup", "unsup"] = "sup"): 
        assert to in ("imagefolder", "hdf5")
        assert subset in ("sup", "unsup") 
    
        if to == "imagefolder":
            ...
            # move to :local/imagefolder/cdl/images,masks and so on for each subset

        elif to == "hdf5": 
            imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder") 
            hdf5_path = fs.get_new_dir(cls.local / "hdf5")

            args: list = [(hdf5_path, imagefolder_path, x, cls.load('index', 'imagefolder', x)) for x in ("cdl", "nlcd", "corine")]

            with Pool(processes=cpu_count()) as pool:
                pool.starmap(cls._write_to_hdf, args)
    
    @staticmethod
    def _write_to_hdf(hdf5_path: Path, imagefolder_path: Path, name: str, df: pd.DataFrame):
        df.to_hdf(hdf5_path / f"spectral_earth_{name}.h5", key = name, mode = 'w')

        with h5py.File(name = hdf5_path / f"spectral_earth_{name}.h5", mode = 'r+') as f:
            images: h5py.Dataset = f.create_dataset(f"{name}_images", shape = (len(df), 202, 128, 128), dtype = np.int16)
            masks: h5py.Dataset = f.create_dataset(f"{name}_masks", shape = (len(df), 128, 128), dtype = np.int16)

            for idx, row in tqdm(df.iterrows(), total = len(df), desc = f"encoding {name}"):
                with rio.open(imagefolder_path/"enmap"/f"{row["image_name"]}.tif") as raster:
                    images[idx] = raster.read()
                with rio.open(imagefolder_path/name/f"{row["image_name"]}.tif") as raster:
                    masks[idx] = raster.read().squeeze()

class SpectralEarth_CDL_Segmentation_HDF5(Dataset): ...
class SpectralEarth_NLCD_Segmentation_HDF5(Dataset): ...
class SpectralEarth_CORINE_Segmentation_HDF5(Dataset): ...