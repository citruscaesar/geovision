from typing import Literal, Optional 

import torch
import tarfile
import shutil
import h5py # type: ignore
import numpy as np
import pandas as pd
import pandera as pa
import imageio.v3 as iio
import torchvision.transforms.v2 as T # type: ignore

from pathlib import Path

from tqdm import tqdm 
from io import BytesIO
from .dataset import Dataset
from .config import DatasetConfig
from .utils import get_valid_split, get_df, get_split_df
from geovision.io.local import get_valid_file_err, get_valid_dir_err, get_new_dir, is_valid_file
from geovision.io.remote import HTTPIO 

import logging
logger = logging.getLogger(__name__)

class Imagenette:
    local = Path.home() / "datasets" / "imagenette"
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    class_names = (
        'tench', 'english_springer', 'cassette_player', 'chain_saw', 'church', 
        'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute'
    )
    num_classes = len(class_names) 
    means = (0.485, 0.456, 0.406)
    std_devs = (0.229, 0.224, 0.225)
    default_config = DatasetConfig(
        random_seed = 42,
        test_sample = 0.2,
        val_sample = 0.1,
        tabular_sampling = "imagefolder_notest",
        image_pre = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale = True), T.Resize(224, antialias=True)]),
        target_pre = T.Identity(),
        train_aug = T.RandomHorizontalFlip(0.5),
        eval_aug = T.Identity() 
    )
    
    @classmethod
    def download(cls):
        r"""download and save imagenette2.tgz to :imagenette/archives/"""
        HTTPIO.download_url(cls.url, cls.local/"archives")
    
    @classmethod
    def transform_to_imagefolder(cls) -> None:
        """extracts files from archive to :imagenette/imagefolder, raises OSError if :imagenette/archives/imagenette2.tgz not found"""

        def move_tempfiles_to_imagefolder(split: Literal["train", "val"]):
            for src_path in tqdm(list((temp_path/"imagenette2"/split).rglob("*.JPEG")), desc = f"{imagefolder_path/split}"):
                dst_path = imagefolder_path/split/src_path.parent.stem/f"{src_path.stem}.jpg"
                dst_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.move(src_path, dst_path)

        imagefolder_path = get_new_dir(cls.local/"imagefolder") 
        temp_path = get_new_dir(cls.local/"temp") 
        with tarfile.open(get_valid_file_err(cls.local/"archives"/"imagenette2.tgz")) as tf:
            tf.extractall(temp_path)
        move_tempfiles_to_imagefolder("train")
        move_tempfiles_to_imagefolder("val")
        shutil.rmtree(temp_path)
        
    @classmethod 
    def transform_to_hdf(cls) -> None:
        """encodes files from imagefolder to :root/hdf5/imagenette.h5, raises OSError if imagenette/imagefolder/images is invalid"""
        imagefolder_path = get_valid_dir_err(cls.local/"imagefolder", empty_ok=False) 
        h5_path = get_new_dir(cls.local/"hdf5") / "imagenette.h5"

        df = cls.get_dataset_df_from_imagefolder()  
        df.to_hdf(h5_path, mode = "w", key = "df")
        with h5py.File(h5_path, mode = "r+") as h5file:
            images = h5file.create_dataset(name = "images", shape = (len(df),), dtype = h5py.special_dtype(vlen = np.dtype('uint8')))
            for idx, row in tqdm(df.iterrows(), total = 13394):
                with open(imagefolder_path/row["image_path"], "rb") as image_file:
                    images[idx] = np.frombuffer(image_file.read(), dtype = np.uint8)

    @classmethod
    def get_dataset_df_from_archive(cls) -> pd.DataFrame:
        """generates and returns dataset_df from :imagenette/archive/imagenette2.tgz, raises OSError if archive is not found""" 
        archive = get_valid_file_err(cls.local/"archives"/"imagenette2.tgz")
        with tarfile.open(archive) as a:
            return pd.DataFrame({"image_path": [Path(p) for p in a.getnames() if p.endswith(".JPEG")]}).pipe(cls._get_dataset_df)

    @classmethod
    def get_dataset_df_from_imagefolder(cls) -> pd.DataFrame:
        """generates and returns dataset_df from :imagenette/imagefolder, raises OSError if imagefolder dir is invalid""" 
        return pd.DataFrame({"image_path": list(get_valid_dir_err(cls.local/"imagefolder", empty_ok=False).rglob("*.jpg"))}).pipe(cls._get_dataset_df)
    
    @classmethod
    def get_dataset_df_from_hdf5(cls) -> pd.DataFrame:
        """returns imagenette dataset_df stored in :imagenette/hdf5/imagenette.h5/df, raises OSError if h5 file is not found, and KeyError if df is not found in h5"""
        return pd.read_hdf(get_valid_file_err(cls.local/"hdf5"/"imagenette.h5"), key = "df", mode = 'r') # type: ignore

    @classmethod 
    def _get_dataset_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        class_synsets = sorted(df["image_path"].apply(lambda x: x.parent.stem).unique())
        return (
            df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
            .assign(label_str = lambda df: df["image_path"].apply(lambda x: x.parent.stem))
            .sort_values("label_str")
            .assign(label_idx = lambda df: df["label_str"].apply(lambda x: class_synsets.index(x)))
            .assign(label_str = lambda df: df["label_idx"].apply(lambda x: cls.class_names[x]))
            .reset_index(drop = True)
        )
    
class ImagenetteImagefolderClassification(Dataset):
    name = "imagenette_imagefolder_classification" 
    class_names = Imagenette.class_names
    num_classes = Imagenette.num_classes 
    means = Imagenette.means
    std_devs = Imagenette.std_devs
    df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),  
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
        "label_str": pa.Column(str, pa.Check.isin(Imagenette.class_names)),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    }, index = pa.Index(int, unique=True))

    split_df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True, checks = [pa.Check(lambda x: is_valid_file(x), element_wise=True)]),
        "df_idx": pa.Column(int, unique = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
    }, index = pa.Index(int, unique=True))

    def __init__(self, split: Literal["train", "val", "trainval", "test", "all"] = "all", config: Optional[DatasetConfig] = None) -> None:
        self._root = get_valid_dir_err(Imagenette.local/"imagefolder") 
        self._split = get_valid_split(split) 
        self._config = config or Imagenette.default_config
        logger.info(f"init {self.name}[{self._split}]\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}")
        self._df = get_df(self._config, self.df_schema, Imagenette.get_dataset_df_from_imagefolder())
        self._split_df = get_split_df(self._df, self.split_df_schema, self._split, self._root)

    def __len__(self) :
        return len(self._split_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        image = iio.imread(idx_row["image_path"], format_hint=".jpg").squeeze()
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        if self._config.image_pre is not None:
            image = self._config.image_pre(image)
        if self._split == "train" and self._config.train_aug is not None:
            image = self._config.train_aug(image) 
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"] 

    @property
    def root(self):
        return self._root

    @property
    def split(self) -> Literal["train", "val", "trainval", "test", "all"]:
        return self._split

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def split_df(self) -> pd.DataFrame:
        return self._split_df

class ImagenetteHDF5Classification(Dataset):
    name = "imagenette_hdf5_classification" 
    class_names = Imagenette.class_names
    num_classes = Imagenette.num_classes 
    means = Imagenette.means
    std_devs = Imagenette.std_devs
    df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
        "label_str": pa.Column(str),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    }, index = pa.Index(int, unique=True))
    split_df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "df_idx": pa.Column(int),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
    }, index = pa.Index(int, unique=True))

    def __init__(self, split: Literal["train", "val", "trainval", "test", "all"] = "all", config: Optional[DatasetConfig] = None) -> None:
        self._root = get_valid_file_err(Imagenette.local/"hdf5"/"imagenette.h5") 
        self._split = get_valid_split(split) 
        self._config = config or Imagenette.default_config
        logger.info(f"init {self.name}[{self._split}] using\nimage_pre = {self._config.image_pre}\ntarget_pre = {self._config.target_pre}\ntrain_aug = {self._config.train_aug}\neval_aug = {self._config.eval_aug}")
        self._df = get_df(self._config, self.df_schema, Imagenette.get_dataset_df_from_hdf5())
        self._split_df = get_split_df(self._df, self.split_df_schema, self._split)

    def __len__(self) -> int:
        return len(self._split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        with h5py.File(self._root, mode = "r") as hdf5_file:
            image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        if self._config.image_pre is not None:
            image = self._config.image_pre(image)
        if self._split == "train" and self._config.train_aug is not None:
            image = self._config.train_aug(image) 
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

    @property
    def root(self):
        return self._root

    @property
    def split(self) -> Literal["train", "val", "trainval", "test", "all"]:
        return self._split

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def split_df(self) -> pd.DataFrame:
        return self._split_df

if __name__ == "__main__":
    Imagenette.download()
    Imagenette.transform_to_imagefolder()
    Imagenette.transform_to_hdf()
