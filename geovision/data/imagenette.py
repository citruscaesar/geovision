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
from geovision.data.dataset import Dataset, SPLITS
from geovision.config import DataFrameConfig, TransformConfig
from geovision.io.local import is_hdf5_file, is_valid_file, is_archive_file, is_valid_dir, is_empty_dir, get_valid_dir
from geovision.io.remote import HTTPIO 

class Imagenette:
    name = "imagenette"
    task = "classification"
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    default_imagefolder_root = str(Path.home() / "datasets" / "imagenette")
    default_hdf5_root = str(Path.home() / "datasets" / "imagenette / imagenette.h5")
    class_labels = {
        'n02979186': 'cassette_player',
        'n03000684': 'chain_saw',
        'n03028079': 'church',
        'n02102040': 'english_springer',
        'n03394916': 'french_horn',
        'n03417042': 'garbage_truck',
        'n03425413': 'gas_pump',
        'n03445777': 'golf_ball',
        'n03888257': 'parachute',
        'n01440764': 'tench',
    }
    means = (0.485, 0.456, 0.406)
    std_devs = (0.229, 0.224, 0.225)
    default_transform  = TransformConfig(
        image_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale = True)]),
        common_transform = T.Resize((224, 224), antialias=True)
    ) 
    hdf5_dataset_shape = (13394, 600, 600, 3) 
    class_names = tuple(sorted(class_labels.values()))
    num_classes = len(class_names) 

    @classmethod
    def download(cls, root: str | Path):
        r"""download and save imagenette2.tgz to :root/archives/"""
        root = Path(root) / "archives"
        HTTPIO.download_url(cls.url, root)
    
    @classmethod
    def transform_to_imagefolder(cls, root: str | Path) -> None:
        """extracts files from imagenette2.tgz to :root/imagefolder
        Parameters
        -
        :root -> points to imagenette root, usually ~/datasets/imagenette
        """
        archive = cls._validate_and_get_archive(Path(root) / "archive" / "imagenette2.tgz")
        tempfolder = get_valid_dir(root, "temp")
        imagefolder = get_valid_dir(root, "imagefolder", "images")
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
        imagefolder_path = cls._validate_and_get_imagefolder(root/"imagefolder")
        hdf5_path = get_valid_dir(root, "hdf5") / "imagenette.h5"

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
    def get_dataset_df_from_archive(cls, archive: str | Path) -> pd.DataFrame:
        r"""returns imagenette dataset_df generated through :archive 

        Parameters
        -
        :root -> path to dataset archive, usually ~/datasets/imagenette/archives/imagenette2.tgz
        Raises
        -
        OSError -> :archive invalid path
        """
        archive = cls._validate_and_get_archive(archive) 
        with tarfile.open(archive) as a:
            return (
                pd.DataFrame({"image_path": [Path(p) for p in a.getnames() if p.endswith(".JPEG")]})
                .pipe(cls._get_dataset_df)
            )

    @classmethod
    def get_dataset_df_from_imagefolder(cls, imagefolder: str | Path) -> pd.DataFrame:
        """generates and returns imagenette dataset_df by looking through the imagefolder

        Parameters
        -
        :imagefolder -> points to imagefolder, usually at ~/datastets/imagenette/imagefolder 
        Raises
        -
        OSError -> :imagefolder invalid path 
        """
        imagefolder = cls._validate_and_get_imagefolder(imagefolder)        
        return pd.DataFrame({"image_path": list(imagefolder.rglob("*.jpg"))}).pipe(cls._get_dataset_df)
    
    @classmethod
    def get_dataset_df_from_hdf5(cls, hdf5_path: str | Path) -> pd.DataFrame:
        """returns imagenette dataset_df stored in :root/hdf5/imagenette.h5["df"]

        Parameters
        -
        :hdf5_path -> points to hdf5 file, usually ~/datasets/imagenette/hdf5/imagenette.h5 

        Raises
        -
        OSError -> :hdf5_path invalid path
        """
        hdf5_path = cls._validate_and_get_hdf5(hdf5_path)     
        return pd.read_hdf(hdf5_path, key = "df", mode = 'r') # type: ignore

    @classmethod 
    def _get_dataset_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df
            .assign(image_path = lambda df: df.image_path.apply(
                lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
            .assign(label_str = lambda df: df.image_path.apply(
                lambda x: cls.class_labels[x.parent.stem].lower().replace(' ', '_')))
            .assign(label_idx = lambda df: df["label_str"].apply(
                lambda x: cls.class_names.index(x)))
            .sort_values("label_idx").reset_index(drop = True)
        )

    @staticmethod
    def _validate_and_get_archive(archive: str | Path) -> Path:
        if not (is_valid_file(archive) and is_archive_file(archive)):
            raise OSError(f"{archive} not found") 
        return Path(archive)

    @classmethod
    def _validate_and_get_imagefolder(cls, imagefolder: str | Path) -> Path:
        if not(is_valid_dir(imagefolder) and not is_empty_dir(imagefolder)):
            raise OSError(f"{imagefolder} not found")
        return Path(imagefolder)

    @classmethod
    def _validate_and_get_hdf5(cls, hdf5_path: str | Path) -> Path:
        if not(is_valid_file(hdf5_path) and is_hdf5_file(hdf5_path)):
            raise OSError(f"{hdf5_path} not found")
        return Path(hdf5_path)
    
class ImagenetteImagefolderClassification(Dataset):
    _df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
        "label_str": pa.Column(str),
        "split": pa.Column(str, pa.Check.isin(SPLITS))
    }, index = pa.Index(int, unique=True))

    _split_df_schema = pa.DataFrameSchema({
        "df_idx": pa.Column(int),
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
    }, index = pa.Index(int, unique=True))

    _default_df_config = DataFrameConfig(
        random_seed = 42,
        test_frac = 0.2,
        val_frac = 0.1,
        splitting_strategy = "stratified"
    )

    def __init__(
            self, 
            root: Path,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            df: Optional[pd.DataFrame | Path] = None,
            df_config: Optional[DataFrameConfig] = None,
            transform: Optional[TransformConfig] = None,
            **kwargs
        ) -> None:
        self._validate_and_set_root_dir(root/"imagefolder")
        self._validate_and_set_split(split)
        self._validate_and_set_transform(transform, Imagenette.default_transform)
        self._validate_and_set_df(
            df = df,
            default_df = Imagenette.get_dataset_df_from_imagefolder(self.root),
            split_col = "label_str",
            config = df_config,
            default_config = self._default_df_config,
            schema = self._df_schema
        )
        self._validate_and_set_imagefolder_split_df(
            split = self.split, 
            root = self.root, 
            schema = self._split_df_schema
        )

    def __len__(self) :
        return len(self._split_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        image = iio.imread(idx_row["image_path"]).squeeze()
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        if self._image_transform is not None:
            image = self._image_transform(image)
        if self._common_transform is not None:
            image = self._common_transform(image)
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
    def name(self) -> str:
        return Imagenette.name 
    
    @property
    def task(self) -> str:
        return Imagenette.task 

    @property
    def class_names(self) -> tuple[str, ...]:
        return Imagenette.class_names

    @property
    def means(self) -> tuple[float, ...]:
        return Imagenette.means

    @property
    def std_devs(self) -> tuple[float, ...]:
        return Imagenette.std_devs

class ImagenetteHDF5Classification(Dataset):
    _df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
        "label_str": pa.Column(str),
        "split": pa.Column(str, pa.Check.isin(SPLITS))
    }, index = pa.Index(int, unique=True))

    _split_df_schema = pa.DataFrameSchema({
        "df_idx": pa.Column(int),
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
    }, index = pa.Index(int, unique=True))

    _default_df_config = DataFrameConfig(
        random_seed = 42,
        test_frac = 0.2,
        val_frac = 0.1,
        splitting_strategy = "stratified"
    )

    def __init__(
            self, 
            root: Path,
            df: Optional[pd.DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            df_config: Optional[DataFrameConfig] = None,
            transform: Optional[TransformConfig] = None,
            **kwargs
        ) -> None:
        self._validate_and_set_root_hdf5(root/"hdf5"/"imagenette.h5")
        self._validate_and_set_split(split)
        self._validate_and_set_transform(transform, Imagenette.default_transform)
        self._validate_and_set_df(
            df = df,
            default_df = Imagenette.get_dataset_df_from_hdf5(self.root),
            split_col = "label_str",
            config = df_config,
            default_config = self._default_df_config,
            schema = self._df_schema
        )
        self._validate_and_set_hdf5_split_df(self.split, self._split_df_schema)

    def __len__(self) -> int:
        return len(self._split_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.split_df.iloc[idx]
        with h5py.File(self.root, mode = "r") as hdf5_file:
            image = iio.imread(BytesIO(hdf5_file["images"][idx_row["df_idx"]]))
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        if self._image_transform is not None:
            image = self._image_transform(image)
        if self._common_transform is not None:
            image = self._common_transform(image)
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
    def name(self) -> str:
        return Imagenette.name 
    
    @property
    def task(self) -> str:
        return Imagenette.task 

    @property
    def class_names(self) -> tuple[str, ...]:
        return Imagenette.class_names

    @property
    def means(self) -> tuple[float, ...]:
        return Imagenette.means

    @property
    def std_devs(self) -> tuple[float, ...]:
        return Imagenette.std_devs