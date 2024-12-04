from typing import Literal, Optional
from numpy.typing import NDArray

import os
import h5py
import py7zr
import shutil
import zipfile
import subprocess
import numpy as np
import pandas as pd
import multivolumefile
import rasterio as rio
import imageio.v3 as iio
import torchvision.transforms.v2 as T

from tqdm import tqdm
from pathlib import Path
from geovision.data.interfaces import Dataset
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url

# NOTE:
# - Extract (from wherever) -> Imagefolder
# - Download (from s3 storage) -> HDF5
# - Load (from imagefolder or HDF5) -> DataFrame
# - Transform (from Imagefolder) -> HDF5
#   -> Calls Load (from imagefolder) -> Index DataFrame

class Inria:
    local = Path.home() / "datasets" / "inria"
    class_names = ("background", "building_rooftop") 
    identity_matrix = np.eye(2, dtype = np.float32)

    @classmethod
    def extract(cls, src: Literal["inria.fr", "huggingface", "kaggle"]):
        """downloads from :src to :local/imagefolder"""
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

            # extract imagefolder from NEW2-AerialImageDataset and change directory structure 
            with zipfile.ZipFile(staging_path / "NEW2-AerialImageDataset.zip", mode = 'r') as archive:
                archive.extractall(imagefolder_path)
            shutil.rmtree(staging_path, ignore_errors=True)
            shutil.move(imagefolder_path / "AerialImageDataset" / "test" / "images", imagefolder_path / "unsup")
            shutil.move(imagefolder_path / "AerialImageDataset" / "train" / "images", imagefolder_path / "images")
            shutil.move(imagefolder_path / "AerialImageDataset" / "train" / "gt", imagefolder_path / "masks")
            shutil.rmtree(imagefolder_path / "AerialImageDataset")
        else:
            raise NotImplementedError 

    @classmethod
    def download(cls, subset: Literal["supervised", "unsupervised"] = "supervised"):
        """downloads inria_:subset.h5 from s3://inria/hdf5/ to :local/hdf5. raises FileNotFoundError if s5cmd not installed in environment or file not in :remote"""
        assert subset in ("supervised", "unsupervised"), f"value error, expected :subset to be one of supervised or unsupervised, got {subset}"
        remote = f"s3://inria/hdf5/inria_{subset}.h5"
        ret = subprocess.call(["s5cmd", "cp", remote, str(fs.get_new_dir(cls.local, "hdf5"))])
        if ret != 0:
            raise FileNotFoundError(f"couldn't find {remote}")

    @classmethod
    def transform(cls, to: Literal["hdf5"] = "hdf5", subset: Literal["supervised", "unsupervised"] = "supervised", transforms: Optional[T.Transform] = None):
        """transforms the :subset of dataset to :to and applies :transforms to each (image, mask) if provided"""
        assert to in ("hdf",), "value error"
        assert subset in ("supervised", "unsupervised"), "value error"
        assert transforms is None, "not implemented yet"

        imagefolder_path = cls.local / "imagefolder"
        index_df = cls.load("index", "imagefolder", subset)
        spatial_df = cls.load("index", "imagefolder", subset)

        hdf5_path = cls.local / "hdf5" / f"inria_{subset}.h5"
        index_df.to_hdf(hdf5_path, key = "index", mode = 'w')
        spatial_df.to_hdf(hdf5_path, key = "spatial", mode = 'r+')

        with h5py.File(hdf5_path, mode = 'r+') as file:
            images = file.create_dataset("images", (180, 5000, 5000, 3), dtype = np.uint8)
            if subset == "supervised":
                masks = file.create_dataset("masks", 180, dtype = h5py.special_dtype(vlen=np.int64))
            for idx, row in tqdm(index_df.iterrows(), total = 180):
                images[idx] = iio.imread(imagefolder_path / row["image_path"], extension=".tif")
                if subset == "supervised":
                    masks[idx] = cls.encode_binary_mask(iio.imread(imagefolder_path / row["mask_path"], extension=".tif"))
 
    @classmethod
    def load(cls, table: Literal["index", "spatial"], src: Literal["imagefolder", "hdf5"], subset: Literal["supervised", "unsupervised"]) -> pd.DataFrame:
        assert src in ("imagefolder", "hdf5")
        assert subset in ("supervised", "unsupervised")
        # check table in metadata file
        if src == "hdf5":
            return pd.read_hdf(cls.local / "hdf5" / f"inria_{subset}.h5", key = table, mode = 'r')
        elif src == "imagefolder":
            metadata_path = cls.local / src / "metadata.h5" 
            try:
                df = pd.read_hdf(metadata_path, key = table, mode = 'r')
            except OSError:
                assert table in ("index", "spatial"), \
                    f"not implemented error, expected :table to be one of index or spatial, since this function cannot create {table} metadata"
                table = {k: list() for k in ("image_path", "xoff", "yoff", "crs")}
                image_paths = (fs.get_valid_dir_err(metadata_path.parent) / ("images" if subset == "supervised" else "unsup")).rglob("*.tif")
                for image_path in tqdm(image_paths, total = 180):
                    table["image_path"].append(f"{image_path.parent.name}/{image_path.name}")
                    with rio.open(image_path) as raster:
                        table["xoff"].append(raster.transform.xoff)
                        table["yoff"].append(raster.transform.yoff)
                        table["crs"].append(raster.crs.to_epsg())

                df = pd.DataFrame(table).sort_values("image_path").reset_index(drop = True)
                if subset == "supervised":
                    df["mask_path"] = df["image_path"].apply(lambda x: f"masks/{x.split('/')[-1]}")
                    df[["image_path", "mask_path"]].to_hdf(metadata_path, key = "index", mode = 'a', index = True)
                elif subset == "unsupervised":
                    df["image_path"] = df["image_path"].apply(lambda x: x.split('/')[-1])
                    df[["image_path"]].to_hdf(metadata_path, key = "index", mode = 'a', index = True)
                df[["xoff", "yoff", "crs"]].to_hdf(metadata_path, key = "spatial", mode = 'r+', index = True)
            return df

    @staticmethod
    def encode_binary_mask(mask : NDArray):
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

class InriaImagefolderSegmentation(Dataset):
    def __init__(self):
        pass

class InriaHDF5Segmentation(Dataset):
    def __init__(self):
        pass