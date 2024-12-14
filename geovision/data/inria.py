from typing import Literal, Optional, Callable
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
import imageio.v3 as iio
import torchvision.transforms.v2 as T

from tqdm import tqdm
from pathlib import Path
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url

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
    def download(cls, subset: Literal["inria"], supervised: bool = True):
        """downloads inria_:subset.h5 from s3://inria/hdf5/ to :local/hdf5. raises FileNotFoundError if s5cmd not installed in environment or file not in :remote"""
        assert subset in ("inria",), f"value error, expected :subset to be one of supervised or unsupervised, got {subset}"
        remote = f"s3://inria/hdf5/{subset}_{"supervised" if supervised else "unsupervised"}.h5"
        ret = subprocess.call(["s5cmd", "cp", remote, str(fs.get_new_dir(cls.local, "hdf5"))])
        if ret != 0:
            raise FileNotFoundError(f"couldn't find {remote}")

    @classmethod
    def load(cls, table: Literal["index", "spatial"], src: Literal["imagefolder", "hdf5"], subset: Literal["inria"], supervised: bool = True) -> pd.DataFrame:
        assert src in ("imagefolder", "hdf5")
        assert subset == "inria" 
        # check table in metadata file
        if src == "hdf5":
            return pd.read_hdf(cls.local / "hdf5" / f"inria_{"supervised" if supervised else "unsupervised"}.h5", key = table, mode = 'r')
        elif src == "imagefolder":
            metadata_path = cls.local / src / "metadata.h5" 
            try:
                return pd.read_hdf(metadata_path, key = table, mode = 'r')
            except OSError:
                assert table in ("index", "spatial"), \
                    f"not implemented error, expected :table to be one of index or spatial, since this function cannot create {table} metadata"
                metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}
                image_paths = (fs.get_valid_dir_err(metadata_path.parent) / ("images" if supervised else "unsup")).rglob("*.tif")
                for image_path in tqdm(image_paths, total = 180, desc = "reading images"):
                    metadata["image_path"].append(f"{image_path.parent.name}/{image_path.name}")
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
                    df["mask_path"] = df["image_path"].apply(lambda x: f"masks/{x.split('/')[-1]}")
                    index_df = df[["image_path", "mask_path", "location"]]
                    index_df.to_hdf(metadata_path, key = "index", mode = 'a', index = True)
                else:
                    df["image_path"] = df["image_path"].apply(lambda x: x.split('/')[-1])
                    index_df = df[["image_path", "location"]]
                    index_df.to_hdf(metadata_path, key = "index", mode = 'a', index = True)

                spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]
                spatial_df.to_hdf(metadata_path, key = "spatial", mode = 'r+', index = True)

                if table == "index":
                    return index_df
                else:
                    return spatial_df

    @classmethod
    def transform(
            cls, 
            to: Literal["hdf5"], 
            subset: Literal["inria"], 
            tile_size: Optional[int | tuple[int, int]] = None,
            tile_stride: Optional[int | tuple[int, int]] = None,
            supervised: bool = True
        ):
        """transforms the :subset of dataset to :to and applies :transforms to each (image, mask) if provided"""
        assert to == "hdf5"
        assert subset == "inria"

        hdf5_path = fs.get_new_dir(cls.local, "hdf5") / f"inria_{"supervised" if supervised else "unsupervised"}.h5"
        imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder")
        index_df = cls.load("index", "imagefolder", subset)
        spatial_df = cls.load("index", "imagefolder", subset)
                    
        if tile_size is not None and tile_stride is not None:
            index_columns = index_df.columns
            index_df = DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
            index_df = index_df.merge(spatial_df, how = 'left', left_index=True, right_index=True)
            index_df["x_off"] = index_df["x_off"] + index_df["tile_tl_0"] * index_df["x_res"]
            index_df["y_off"] = index_df["y_off"] + index_df["tile_tl_1"] * index_df["y_res"]
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            index_df["image_height"] = tile_size[0]
            index_df["image_width"] = tile_size[1]

            with h5py.File(hdf5_path, mode = 'w') as file:
                images = file.create_dataset("images", (len(index_df), *tile_size, 3), dtype = np.uint8)
                images.attrs["image_size"] = tile_size
                if supervised:
                    masks = file.create_dataset("masks", len(index_df), dtype=h5py.special_dtype(vlen=np.uint8))
                for idx, row in tqdm(index_df.iterrow(), total = len(index_df)):
                    image = iio.imread(imagefolder_path / row["image_path"], extension=".tif")
                    image = image[row["tile_tl_0"]: min(row["tile_br_0"], 5000), row["tile_tl_1"]: min(row["tile_br_1"], 5000), :]

                    if supervised:
                        mask = iio.imread(imagefolder_path / row["mask_path"], extension=".tif").squeeze()
                        mask = mask[row["tile_tl_0"]: min(row["tile_br_0"], 5000), row["tile_tl_1"]: min(row["tile_br_1"], 5000)]

                    if image.shape[0] < tile_size[0] or image.shape[1] < tile_size[1]:
                        pad_along_0 = tile_size[0] - image.shape[0]
                        if pad_along_0 % 2 == 0:
                            pad_along_0 = (pad_along_0//2, pad_along_0//2)
                        else:
                            pad_along_0 = (pad_along_0//2, (pad_along_0//2) + 1)

                        pad_along_1 = tile_size[1] - image.shape[1]
                        if pad_along_1 % 2 == 0:
                            pad_along_1 = (pad_along_1//2, pad_along_1//2)
                        else:
                            pad_along_1 = (pad_along_1//2, (pad_along_1//2) + 1)

                        image = np.pad(image, (pad_along_0, pad_along_1, 0)) 
                        if supervised:
                            mask = np.pad(mask, (pad_along_0, pad_along_1))
                        
                    images[idx] = image
                    if supervised:
                        masks[idx] = cls.encode_binary_mask(mask)
            
            index_df[index_columns].to_hdf(hdf5_path, key = 'index', mode = 'r+')
            index_df[index_df.columns.difference(index_columns)].to_hdf(hdf5_path, key = 'spatial', mode = 'r+')

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

InriaIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "mask_path": pa.Column(str, coerce=True),
        "location": pa.Column(str, coerce=True),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    },
    index=pa.Index(int, unique=True)
)

class InriaImagefolderSegmentation(Dataset):
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

        self.crop = False # flag to indicate cropping is required 
        if "tile_tl_1" in self.df.columns: 
            self.crop = True
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        image = iio.imread(idx_row["image_path"], extension=".tif")
        mask = iio.imread(idx_row["mask_path"], extension=".tif") # mask: (H, W, 1)
        mask = self.identity_matrix[np.clip(mask.squeeze(), 0, 1)] # mask: (H, W, num_channels)
        #print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        #print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())

        if self.crop:
            # NOTE: might have to explicitly copy and delete arrays after slicing, to avoid memory overflow
            image = image[idx_row["tile_tl_1"]:min(idx_row["tile_br_1"], 5000), idx_row["tile_tl_0"]:min(idx_row["tile_br_0"], 5000), :]
            mask = mask[idx_row["tile_tl_1"]:min(idx_row["tile_br_1"], 5000), idx_row["tile_tl_0"]:min(idx_row["tile_br_0"], 5000), :]
            #if idx_row["tile_br_0"] > 5000 or idx_row["tile_br_1"] > 5000:
                #image = np.pad(image, ((0, max(0, idx_row["tile_br_1"]) - 5000), (0, max(0, idx_row["tile_br_0"] - 5000)), (0, 0)))
                #mask = np.pad(image, ((0, max(0, idx_row["tile_br_1"]) - 5000), (0, max(0, idx_row["tile_br_0"] - 5000))))

        image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        #print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        #print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())
        return image, mask, idx_row["df_idx"]

class InriaHDF5Segmentation(Dataset):
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

        self.crop = False # flag to indicate cropping is required 
        if "tile_tl_1" in self.df.columns: 
            self.crop = True
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode = "r") as f:
            image = f["images"][idx_row["df_idx"]]
            mask = Inria.decode_binary_mask(f["masks"][idx_row["df_idx"]], shape = (5000, 5000)) # mask: (H, W)
        mask = self.identity_matrix[np.clip(mask, 0, 1)] # mask: (H, W, num_channels)
        #print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        #print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())

        if self.crop:
            # NOTE: might have to explicitly copy and delete arrays after slicing, to avoid memory overflow
            image = image[idx_row["tile_tl_1"]:min(idx_row["tile_br_1"], 5000), idx_row["tile_tl_0"]:min(idx_row["tile_br_0"], 5000), :]
            mask = mask[idx_row["tile_tl_1"]:min(idx_row["tile_br_1"], 5000), idx_row["tile_tl_0"]:min(idx_row["tile_br_0"], 5000), :]
            if idx_row["tile_br_0"] > 5000 or idx_row["tile_br_1"] > 5000:
                image = np.pad(image, ((0, max(0, idx_row["tile_br_1"]) - 5000), (0, max(0, idx_row["tile_br_0"] - 5000)), (0, 0)))
                mask = np.pad(image, ((0, max(0, idx_row["tile_br_1"]) - 5000), (0, max(0, idx_row["tile_br_0"] - 5000))))

        image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        #print(image.shape, image.dtype, image.min(), image.max(), image.mean())
        #print(mask.shape, mask.dtype, mask.min(), mask.max(), mask.mean())
        return image, mask, idx_row["df_idx"]