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
from .dataset import Dataset, DatasetConfig, TransformsConfig, Validator
from geovision.io.local import get_valid_file_err, get_valid_dir_err, get_new_dir, is_valid_file
from geovision.io.local import get_dataset_dir
from geovision.io.remote import HTTPIO 
from geovision.logging import get_logger
logger = get_logger("imagenette")

class Imagenette:
    name = "imagenette"
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
        tabular_sampling = "stratified"
    )
    default_transforms  = TransformsConfig(
        image_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale = True), T.Resize((224, 224), antialias=True)]),
        common_transform = T.Identity() 
    ) 

    @classmethod
    def download(cls, root: str | Path):
        r"""download and save imagenette2.tgz to :root/archives/"""
        HTTPIO.download_url(cls.url, get_dataset_dir(root, "archives", invalid_ok=True))
    
    @classmethod
    def transform_to_imagefolder(cls, root: str | Path) -> None:
        """extracts files from imagenette2.tgz to :root/imagefolder
        Parameters
        -
        :root -> points to imagenette root, usually ~/datasets/imagenette
        """
        archive = get_valid_file_err(get_dataset_dir(root, "archives"), "imagenette2.tgz", valid_extns=(".tgz",))
        imagefolder = get_dataset_dir(root, "images") 
        tempfolder = get_dataset_dir(root, "temp") 
        with tarfile.open(archive) as tf:
            tf.extractall(tempfolder)
        for temp_path in tqdm(tempfolder.rglob("*.JPEG"), total = 13394):
            image_path = imagefolder / temp_path.parent.name / f"{temp_path.stem}.jpg"
            image_path.parent.mkdir(exist_ok=True)
            shutil.move(temp_path, image_path)
        shutil.rmtree(tempfolder)
        
    @classmethod 
    def transform_to_hdf(cls, root: Path) -> None:
        """encodes files from imagefolder to :root/hdf5/imagenette.h5
        Parameters
        -
        :root -> points to imagenette root, usually ~/datasets/imagenette
        """
        imagefolder_path = get_dataset_dir(root, "imagefolder", invalid_ok=False) 
        hdf5_path = get_dataset_dir(root, "hdf5", invalid_ok=True) / "imagenette.h5"

        df = cls.get_dataset_df_from_imagefolder(root)  
        df.to_hdf(hdf5_path, mode = "w", key = "df")
        with h5py.File(hdf5_path, mode = "r+") as hdf5_file:
            images = hdf5_file.create_dataset(
                name = "images", 
                shape = (13394,),
                dtype=h5py.special_dtype(vlen = np.dtype('uint8'))
            )
            for idx, row in tqdm(df.iterrows(), total = len(df)):
                with open(imagefolder_path/row["image_path"], "rb") as image_file:
                    image_bytes = image_file.read()
                    images[idx] = np.frombuffer(image_bytes, dtype = np.uint8)

    @classmethod
    def get_dataset_df_from_archive(cls, root: str | Path) -> pd.DataFrame:
        r"""returns imagenette dataset_df generated through :archive 

        Parameters
        -
        :root -> path to dataset root, usually ~/datasets/imagenette

        Raises
        -
        OSError -> :archive invalid path if not found at :root/archives/imagenette2.tgz
        """
        archive = get_valid_file_err(get_dataset_dir(root, "archives"), "imagenette2.tgz", valid_extns=(".tgz",))
        with tarfile.open(archive) as a:
            return (
                pd.DataFrame({"image_path": [Path(p) for p in a.getnames() if p.endswith(".JPEG")]})
                .pipe(cls._get_dataset_df)
            )

    @classmethod
    def get_dataset_df_from_imagefolder(cls, root: str | Path) -> pd.DataFrame:
        """generates and returns imagenette dataset_df by looking through the imagefolder

        Parameters
        -
        :root -> points to root, usually at ~/datastets/imagenette 

        Raises
        -
        OSError -> :imagefolder invalid path if not found at :root/imagefolder/images
        """
        imagefolder = get_dataset_dir(root, "imagefolder") 
        return pd.DataFrame({"image_path": list(imagefolder.rglob("*.jpg"))}).pipe(cls._get_dataset_df)
    
    @classmethod
    def get_dataset_df_from_hdf5(cls, root: str | Path) -> pd.DataFrame:
        """returns imagenette dataset_df stored in :root/hdf5/imagenette.h5["df"]

        Parameters
        -
        :hdf5_path -> points to root, usually ~/datasets/imagenette/ 

        Raises
        -
        OSError -> :hdf5_path invalid path
        """
        hdf5_path = get_valid_file_err(get_dataset_dir(root, "hdf5"), "imagenette.h5", valid_extns=(".h5", ".hdf5"))
        return pd.read_hdf(hdf5_path, key = "df", mode = 'r') # type: ignore

    @classmethod 
    def _get_dataset_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        class_synsets = sorted(df["image_path"].apply(lambda x: x.parent.stem).unique())
        return (
            df
            .assign(image_path = lambda df: df["image_path"].apply(
                lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
            .assign(label_str = lambda df: df["image_path"].apply(
                lambda x: x.parent.stem))
            .sort_values("label_str")
            .assign(label_idx = lambda df: df["label_str"].apply(
                lambda x: class_synsets.index(x)))
            .assign(label_str = lambda df: df["label_idx"].apply(
                lambda x: cls.class_names[x]))
            .assign(split_on = lambda df: df["label_str"])
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
        "image_path": pa.Column(str, coerce = True, checks = [
            pa.Check(lambda x: is_valid_file(x, valid_extns=(".jpg",)), element_wise=True)
        ]),
        "df_idx": pa.Column(int, unique = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
    }, index = pa.Index(int, unique=True))

    def __init__(
            self, 
            root: Path,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            df: Optional[pd.DataFrame | Path] = None,
            config: Optional[DatasetConfig] = None,
            transforms: Optional[TransformsConfig] = None,
            **kwargs
        ) -> None:
        logger.info(f"initializing {self.name}")
        self._root = Validator._get_root_dir(get_dataset_dir(root, "imagefolder"))
        self._split = Validator._get_split(split)
        self._transforms = Validator._get_transforms(transforms, Imagenette.default_transforms)
        self._df = Validator._get_df(
            df = df,
            config = config,
            schema = self.df_schema,
            default_df = Imagenette.get_dataset_df_from_imagefolder(root),
            default_config = Imagenette.default_config,
        )
        self._split_df = Validator._get_imagefolder_split_df(
            df = self._df,
            schema = self.split_df_schema,
            root = self._root,
            split = self._split
        )

    def __len__(self) :
        return len(self._split_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        image = iio.imread(idx_row["image_path"], format_hint=".jpg").squeeze()
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        image = self._transforms.image_transform(image) # type: ignore
        if self._split == "train":
            image = self._transforms.common_transform(image) # type: ignore
        return image, idx_row["label_idx"], idx_row["df_idx"] # type: ignore 

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

    @property
    def transforms(self) -> TransformsConfig:
        return self._transforms

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

    def __init__(
            self, 
            root: Path,
            df: Optional[pd.DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            config: Optional[DatasetConfig] = None,
            transforms: Optional[TransformsConfig] = None,
            **kwargs
        ) -> None:
        logger.info(f"initializing {self.name}")
        self._root = Validator._get_root_hdf5(get_dataset_dir(root, "hdf5")/"imagenette.h5")
        self._split = Validator._get_split(split)
        self._transforms = Validator._get_transforms(transforms, Imagenette.default_transforms)
        self._df = Validator._get_df(
            df = df,
            config = config,
            schema = self.df_schema,
            default_df = Imagenette.get_dataset_df_from_hdf5(root),
            default_config = Imagenette.default_config,
        )
        self._split_df = Validator._get_imagefolder_split_df(
            df = self._df,
            schema = self.split_df_schema,
            root = self._root,
            split = self._split
        )
        
    def __len__(self) -> int:
        return len(self._split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.split_df.iloc[idx]
        with h5py.File(self.root, mode = "r") as hdf5_file:
            image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        image = self._transforms.image_transform(image) # type: ignore
        if self._split == "train":
            image = self._transforms.common_transform(image)
        return image, idx_row["label_idx"], idx_row["df_idx"] # type: ignore

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

    @property
    def transforms(self) -> TransformsConfig:
        return self._transforms