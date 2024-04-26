from typing import Literal, Optional 

import torch
import numpy as np
import pandas as pd
import pandera as pa
import imageio.v3 as iio
import torchvision.transforms.v2 as T # type: ignore

from pathlib import Path 
from geovision.data.dataset import Dataset, SPLITS

from .tests import TestDataset

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
    default_image_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale = True),
        T.Resize((224, 224), antialias=True)
    ]) 
    default_common_transform = T.Identity()
     
    class_names = tuple(sorted(class_labels.values()))
    num_classes = len(class_names) 

    @classmethod
    def get_dataset_df(cls, root: Path) -> pd.DataFrame:
        return (
            pd.DataFrame({"image_path": list(root.rglob("*.JPEG"))})
            .assign(image_path = lambda df: df.image_path.apply(
                lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
            .assign(label_str = lambda df: df.image_path.apply(
                lambda x: cls.class_labels[x.parent.stem].lower().replace(' ', '_')))
            .assign(label_idx = lambda df: df["label_str"].apply(
                lambda x: cls.class_names.index(x)))
            .assign(split = lambda df: df.image_path.apply(
                lambda x: "test" if x.parents[1].stem == "val" else "train"))
        )

    @classmethod 
    def transform_to_hdf(cls, root: Path) -> None:
        pass

class ImagenetteImagefolderClassification(Dataset):
    _df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
        "split": pa.Column(str, pa.Check.isin(SPLITS))
    }, index = pa.Index(int, unique=True))

    _split_df_schema = pa.DataFrameSchema({
        "df_idx": pa.Column(int),
        "image_path": pa.Column(str, coerce = True),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, Imagenette.num_classes)))),
    }, index = pa.Index(int, unique=True))

    def __init__(
            self, 
            root: Path,
            df: Optional[pd.DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            val_split: float = 0.1,
            test_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[T.Transform] = None,
            target_transform: Optional[T.Transform] = None,
            common_transform: Optional[T.Transform] = None,
            **kwargs
        ) -> None:
        self._validate_and_set_root_dir(root)
        self._validate_and_set_split(split)
        self._validate_and_set_image_transform(image_transform, Imagenette.default_image_transform)
        self._validate_and_set_common_transform(common_transform, Imagenette.default_common_transform)
        if not isinstance(df, pd.DataFrame):
            df = self._get_stratified_split_df(
                df = Imagenette.get_dataset_df(self.root),
                split_col = "label_str",
                split_params = self._validate_and_get_split_params(
                    test_split, val_split, random_seed
                )
            )
        self._validate_and_set_df(df, self._df_schema)
        self._validate_and_set_split_df(df, self.split, self.root, self._split_df_schema)

    def __len__(self) :
        return len(self._split_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self._split_df.iloc[idx]
        image = iio.imread(idx_row["image_path"]).squeeze()
        image = np.stack((image,)*3, axis = -1) if image.ndim == 2 else image
        image = self._common_transform(self._image_transform(image))
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
    pass

# from .tests import (  # noqa: E402
    # imagefolder_invalid_root,
    # imagefolder_valid_root,
    # # dataset_invalid_splits,
    # # dataset_valid_splits,
    # setup_imagefolder_tests
# )

# def test_imagefolder_invalid_root():
    # imagefolder_invalid_root(ImagenetteImagefolderClassification)

# setup_imagefolder_tests(ImagenetteImagefolderClassification, Imagenette.default_imagefolder_root)
# def test_imagefolder_valid_root():
    # imagefolder_valid_root()

# def test_imagefolder_invalid_splits():
    # dataset_invalid_splits(
        # dataset = ImagenetteImagefolderClassification, 
        # valid_root = Imagenette.default_imagefolder_root
    # )

# def test_imagefolder_valid_splits():
    # dataset_valid_splits(
        # dataset = ImagenetteImagefolderClassification, 
        # valid_root = Imagenette.default_imagefolder_root
    # )

#def test_imagefolder_invalid_transforms():
    #dataset_invalid_transforms(
        #dataset = ImagenetteImagefolderClassification,
        #valid_root = Imagenette.default_imagefolder_root 
    #)

# def test_imagefolder_valid_splits():
    # dataset_valid_splits(init_imagefolder_ds, Imagenette.default_imagefolder_root)