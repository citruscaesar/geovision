from typing import Literal
from numpy.typing import NDArray

import h5py
import py7zr
import shutil
import zipfile
import numpy as np
import pandas as pd
import multivolumefile
import rasterio as rio
import imageio.v3 as iio

from pathlib import Path
from .dataset import Dataset
from geovision.io.local import get_new_dir, get_valid_dir_err
from torchvision.datasets.utils import download_url
from tqdm import tqdm

class Inria:
    local = Path.home() / "datasets" / "inria"
    archive = local / "archives" / "NEW2-AerialImageDataset.zip"
    urls = (
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
        "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005"
    )
    class_names = ("background", "building_rooftop") 
    means = ()
    std_devs = ()
    identity_matrix = np.eye(2, dtype = np.float32)

    # Extraction
    @classmethod
    def download_from_source(cls):
        archives_dir = get_new_dir(cls.local / "archives") 
        for url in tqdm(cls.urls):
            download_url(url, archives_dir)
        with multivolumefile.open(archives_dir / "aerialimagelabeling.7z", mode = "rb") as multi_archive:
            with py7zr.SevenZipFile(multi_archive, mode = "r") as archive:
                archive.extractall(archives_dir)
        for url in cls.urls:
            (archives_dir / url.split('/')[-1]).unlink(missing_ok=True)

    @classmethod
    def download_from_storage(cls, storage_type: Literal["hdf5", "archive"] = "hdf5", subset: Literal["supervised", "unsupervised", "both"] = "supervised"):
        """call s5cmd sync with index-url and save to appropriate dir"""
        pass

    # Transformation
    @classmethod
    def transform_to_imagefolder(cls):
        imagefolder_path = get_new_dir(cls.local / "imagefolder")
        with zipfile.ZipFile(cls.archive, mode = "r") as archive:
            archive.extractall(imagefolder_path)
        shutil.move(imagefolder_path / "AerialImageDataset" / "test" / "images", imagefolder_path / "unsup")
        shutil.move(imagefolder_path / "AerialImageDataset" / "train" / "images", imagefolder_path / "images")
        shutil.move(imagefolder_path / "AerialImageDataset" / "train" / "gt", imagefolder_path / "masks")
        shutil.rmtree(imagefolder_path / "AerialImageDataset")
        cls.get_dataset_df_from_imagefolder()

    @classmethod
    def transform_to_hdf5_from_archive(cls, subset: Literal["supervised", "unsupervised", "both"] = "supervised"):
        hdf5_path = get_new_dir(cls.local / "hdf5")
        df = cls.get_dataset_df_from_archive()

        if subset in ("supervised", "both"):
            supervised_df = df[df["supervised"]].drop(columns = "supervised").reset_index(drop = True)
            with h5py.File(hdf5_path / "inria_supervised.h5", "w") as h5file:
                images = h5file.create_dataset("images", (180, 5000, 5000, 3), dtype = np.uint8)
                masks = h5file.create_dataset("masks", 180, dtype = h5py.special_dtype(vlen=np.int64))
                for idx, row in tqdm(supervised_df.iterrows(), total = 180, desc = "writing supervised samples"):
                    images[idx] = iio.imread(f"{cls.archive}/{row["image_path"]}", extension=".tif")
                    masks[idx] = cls.encode_binary_mask(iio.imread(f"{cls.archive}/{row["mask_path"]}", extension=".tif"))
            supervised_df.to_hdf(hdf5_path / "inria_supervised.h5", "dataset_df", "r+")
            
        if subset in ("unsupervised", "both"):
            unsupervised_df = df[~df["supervised"]].drop(columns = ["mask_path", "supervised"]).reset_index(drop = True)
            with h5py.File(hdf5_path / "inria_unsupervised.h5", "w") as h5file:
                images = h5file.create_dataset("images", (180, 5000, 5000, 3), dtype = np.uint8)
                for idx, row in tqdm(unsupervised_df.iterrows(), total = 180, desc = "writing unsupervised samples"):
                    images[idx] = iio.imread(f"{cls.archive}/{row["image_path"]}", extension=".tif")
            unsupervised_df.to_hdf(hdf5_path / "inria_unsupervised.h5", "dataset_df", "r+")

    @classmethod
    def transform_to_hdf5_from_imagefolder(cls, subset: Literal["supervised", "unsupervised", "both"] = "supervised"):
        imagefolder_path = get_valid_dir_err(cls.local/"imagefolder")
        hdf5_path = get_new_dir(cls.local / "hdf5")
        df = cls.get_dataset_df_from_imagefolder()

        if subset in ("supervised", "both"):
            supervised_df = df[df["supervised"]].drop(columns = "supervised")
            with h5py.File(hdf5_path / "inria_supervised.h5", "w") as h5file:
                images = h5file.create_dataset("images", (180, 5000, 5000, 3), dtype = np.uint8)
                masks = h5file.create_dataset("masks", 180, dtype = h5py.special_dtype(vlen=np.int64))
                for idx, row in tqdm(supervised_df.iterrows(), total = 180):
                    images[idx] = iio.imread(imagefolder_path / row["image_path"], extension=".tif")
                    masks[idx] = cls.encode_binary_mask(iio.imread(imagefolder_path / row["mask_path"], extension=".tif"))
            supervised_df.to_hdf(hdf5_path / "inria_supervised.h5", "dataset_df", "r+")

        if subset in ("unsupervised", "both"):
            unsupervised_df = df[~df["supervised"]].drop(columns = ["mask_path", "supervised"]).reset_index(drop = True)
            with h5py.File(hdf5_path / "inria_unsupervised.h5", "w") as h5file:
                images = h5file.create_dataset("images", (180, 5000, 5000, 3), dtype = np.uint8)
                for idx, row in tqdm(unsupervised_df.iterrows(), total = 180):
                    images[idx] = iio.imread(imagefolder_path / row["image_path"], extension=".tif")
            unsupervised_df.to_hdf(hdf5_path / "inria_unsupervised.h5", "dataset_df", "r+")

    @classmethod
    def get_dataset_df_from_archive(cls):
        dataset_dict = {"image_path": list(), "xoff": list(), "yoff": list(), "crs": list()}
        with zipfile.ZipFile(cls.archive) as zf:
            image_paths = sorted([x for x in sorted(zf.namelist()) if x.endswith(".tif") and "train/gt/" not in x])
            for image_path in tqdm(image_paths, desc = f"reading metadata from {cls.archive.name}"):
                dataset_dict["image_path"].append(image_path)
                with rio.open(f"zip://{zf.filename}!{image_path}") as raster:
                    dataset_dict["xoff"].append(raster.transform.xoff)
                    dataset_dict["yoff"].append(raster.transform.yoff)
                    dataset_dict["crs"].append(raster.crs.to_epsg())
        df = (
            pd.DataFrame(dataset_dict)
            .assign(supervised = lambda df: df["image_path"].apply(lambda x: x.split('/')[-3] == "train"))
            .assign(mask_path = lambda df: df["image_path"].apply(lambda x: Path(x).parents[1]/"gt"/Path(x).name))
        )
        return df[["image_path", "mask_path", "xoff", "yoff", "crs", "supervised"]].sort_values("image_path").reset_index(drop = True)

    @classmethod
    def get_dataset_df_from_imagefolder(cls) -> pd.DataFrame:
        imagefolder_path = get_valid_dir_err(cls.local / "imagefolder")
        metadata_path = imagefolder_path / "metadata.h5"
        try:
            df = pd.read_hdf(metadata_path, "dataset_df", "r")
        except OSError:
            dataset_dict = {"image_path": list(), "xoff": list(), "yoff": list(), "crs": list()}
            image_paths = list((imagefolder_path/"images").rglob("*.tif")) + list((imagefolder_path/"unsup").rglob("*.tif"))
            for image_path in tqdm(image_paths):
                dataset_dict["image_path"].append(f"{image_path.parent.name}/{image_path.name}")
                with rio.open(image_path) as raster:
                    dataset_dict["xoff"].append(raster.transform.xoff)
                    dataset_dict["yoff"].append(raster.transform.yoff)
                    dataset_dict["crs"].append(raster.crs.to_epsg())

            df = (
                pd.DataFrame(dataset_dict)
                .assign(supervised = lambda df: df["image_path"].apply(lambda x: x.split('/')[0] == "images"))
                .assign(mask_path = lambda df: df["image_path"].apply(lambda x: f"masks/{x.split('/')[-1]}"))
            )
            df = df[["image_path", "mask_path", "xoff", "yoff", "crs", "supervised"]].sort_values("image_path").reset_index(drop = True)
            df.to_hdf(metadata_path, key = "dataset_df", mode = "w")
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