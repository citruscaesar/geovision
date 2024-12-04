from typing import Callable, Sequence, Literal, Optional, Any
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import logging
import pandas as pd
import pandera as pa
from pathlib import Path
from abc import ABC, abstractmethod 
from torch.utils.data import DataLoader, default_collate
from lightning import LightningDataModule
from litdata import StreamingDataLoader, StreamingDataset
from torchvision.transforms.v2 import Transform, Identity, CutMix, MixUp

from geovision.io.local import FileSystemIO as fs 

logger = logging.getLogger(__name__)

class DatasetConfig:
    def __init__(
            self,
            random_seed: int, 
            df: Optional[str] = None,
            tabular_sampler_name: Optional[str] = None,
            tabular_sampler_params: Optional[dict] = None,
            spatial_sampler_name: Optional[str] = None,
            spatial_sampler_params: Optional[dict] = None,
            spectral_sampler_name: Optional[str] = None,
            spectral_sampler_params: Optional[dict] = None,
            temporal_sampler_name: Optional[str] = None,
            temporal_sampler_params: Optional[dict] = None,
            image_pre: Optional[Transform] = None,
            target_pre: Optional[Transform] = None,
            train_aug: Optional[Transform] = None,
            eval_aug: Optional[Transform] = None,
            **kwargs
    ):
        self.random_seed = random_seed
        if isinstance(df, str | Path):
            self.df_path = fs.get_valid_file_err(df) 
        else:
            self.df_path = None 

        self.tabular_sampler_name = tabular_sampler_name
        if self.tabular_sampler_name is not None:
            self.tabular_sampler = self._get_tabular_sampler(tabular_sampler_name)
            self.tabular_sampler_params = ( tabular_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        else:
            self.tabular_sampler = None
            self.tabular_sampler_params = None

        self.spatial_sampler_name = spatial_sampler_name
        if self.spatial_sampler_name is not None:
            self.spatial_sampler = self._get_spatial_sampler(spatial_sampler_name)
            self.spatial_sampler_params = ( spatial_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        else:
            self.spatial_sampler = None
            self.spatial_sampler_params = None

        self.spectral_sampler_name = spectral_sampler_name
        if self.spectral_sampler_name is not None:
            self.spectral_sampler = self._get_spectral_sampler(spectral_sampler_name)
            self.spectral_sampler_params = ( spectral_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        else:
            self.spectral_sampler = None
            self.spectral_sampler_params = None

        self.temporal_sampler_name = temporal_sampler_name
        if self.temporal_sampler_name is not None:
            self.temporal_sampler = self._get_temporal_sampler(temporal_sampler_name)
            self.temporal_sampler_params = ( temporal_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        else:
            self.temporal_sampler = None
            self.temporal_sampler_params = None

        self.image_pre = image_pre or Identity()
        self.target_pre = target_pre or Identity()
        self.train_aug = train_aug or Identity()
        self.eval_aug = eval_aug or Identity()
    
    def verify_and_get_df(self, schema: pa.DataFrameSchema, fallback_df: pd.DataFrame) -> pd.DataFrame:
        if self.df_path is not None:
            if self.df_path.suffix == ".csv":
                df = pd.read_csv(self.df_path)
            elif self.df_path.suffix == ".h5":
                df = pd.read_hdf(self.df_path, key = "df", mode = "r")
            else:
                raise ValueError(f"unknown suffix for :df, expected one of .csv, .h5, got {self.df_path.suffix}")
        else:
            df = fallback_df
            for sampler in ("tabular", "spatial", "spectral", "temporal"):
                sampler_fn = self.__getattribute__(f"{sampler}_sampler")
                params = self.__getattribute__(f"{sampler}_sampler_params")
                if sampler_fn is not None:
                    logger.info(f"applying {sampler} sampling, using fn: {sampler_fn} and params: {params}")
                    df = sampler_fn(df = df, **params)
                else:
                    logger.info(f"did not find {sampler} sampler, skipping")
        return schema(df)
    
    def verify_and_get_split_df(self, df: pd.DataFrame, schema: pa.DataFrameSchema, split: str, prefix_root: Optional[Path] = None) -> pd.DataFrame:
        """returns a (re-indexed) subset of :df with rows containing :split columns, and performs a schema check at the end. raises value error if :split is invalid,
        schema error if df does not have a column named split or split_df fails schema check"""

        split_df = df.assign(df_idx = lambda df: df.index)
        assert "split" in split_df.columns, "schema error, dataframe does not have a column named 'split'"
        assert split in Dataset.splits, f"value error, expected split to be one of {Dataset.splits}, got {split}"

        if split not in ("all", "trainvaltest"):
            split_df = split_df[split_df.split == split].reset_index(drop=True)

        if prefix_root is not None:
            if "image_path" in split_df.columns:
                split_df["image_path"] = split_df["image_path"].apply(lambda x: str(Path(prefix_root, x)))
            if "mask_path" in split_df.columns:
                split_df["mask_path"] = split_df["mask_path"].apply(lambda x: str(Path(prefix_root, x)))

        return schema(split_df)

    def _get_tabular_sampler(self, tabular_sampler_name: str):  # noqa: F821
        if tabular_sampler_name == "stratified":
            return _get_stratified_split_df
        elif tabular_sampler_name ==  "imagefolder_notest":
            return _get_imagefolder_notest_split_df
        else:
            raise AssertionError(f"not implemented error, {tabular_sampler_name} has not been implemented/registered yet")
    
    def _get_spatial_sampler(self, spatial_sampler_name: str):
        return None

    def _get_spectral_sampler(self, spectral_sampler_name: str):
        return None

    def _get_temporal_sampler(self, temporal_sampler_name: str):
        return None
        
    def __repr__(self) -> str:
        out = f"DataFrame Path: [{self.df_path}]\n"
        for category in ("tabular", "spatial", "spectral", "temporal"):
            sampler_name, sampler_params = f"{category}_sampler_name", f"{category}_sampler_params"
            if hasattr(self, sampler_name):
                out += f"{' '.join(map(lambda x: x.capitalize(), sampler_name.split('_')))}: {self.__getattribute__(sampler_name)} [{self.__getattribute__(sampler_name.removesuffix("_name"))}]\n" 
                out += f"{' '.join(map(lambda x: x.capitalize(), sampler_params.split('_')))}: {self.__getattribute__(sampler_params)}\n"
        out += f"Preprocess Image: {self.image_pre}\n"
        out += f"Preprocess Target: {self.target_pre}\n"
        out += f"Train-Time Augmentations: {self.train_aug}\n"
        out += f"Eval-Time Agumentations: {self.eval_aug}\n"
        return out

class DataLoaderConfig:
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            persistent_workers: Optional[bool] = None,
            pin_memory: Optional[bool] = None,
            prefetch_factor: Optional[int] = None,
            gradient_accumulation: int = 1,
            batch_transform_name: Optional[str] = None,
            batch_transform_params: Optional[dict] = None,
            **kwargs
    ):
        self.batch_size = batch_size // gradient_accumulation
        self.gradient_accumulation = gradient_accumulation

        self.collate_fn: Optional[Callable] = None
        if batch_transform_name is not None:
            assert batch_transform_name in ("cutmix", "mixup"), f"config error (not implemented), expected :batch_transform_name to be one of cutmix, mixup or None, got {batch_transform_name}"
            if batch_transform_name == "cutmix":
                self.cutmix = CutMix(**batch_transform_params)
                self.collate_fn = lambda batch: self.cutmix(*default_collate(batch))
            else:
                self.mixup = MixUp(**batch_transform_params)
                self.collate_fn = lambda batch: self.mixup(*default_collate(batch))

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    @property
    def params(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "collate_fn": self.collate_fn
        }

class Dataset(ABC):
    splits = ("train", "val", "test", "trainvaltest", "all")

    @abstractmethod
    def __init__(self, split: str = "all", config: Optional[DatasetConfig] = None) -> None:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def root(self) -> Path:
        pass

    @property
    @abstractmethod
    def split(self) -> str:
        pass

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def split_df(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def means(self) -> tuple[float, ...]:
        pass

    @property
    @abstractmethod
    def std_devs(self) -> tuple[float, ...]:
        pass

    def __repr__(self) -> str:
        name, storage, task = self.name.split('_')
        return '\n'.join([
            f"{name} dataset for {task}",
            f"local {storage} @ [{self.root}] ",
            f"with {len(self.class_names)} classes and {len(self)} images under the '{self.split}' split",
        ])

    @property
    def num_total_samples(self) -> int: 
        return len(self.df)

    @property
    def num_train_samples(self) -> int: 
        return len(self.df[self.df["split"]=="train"])

    @property
    def num_val_samples(self) -> int: 
        return len(self.df[self.df["split"]=="val"])

    @property
    def num_test_samples(self) -> int: 
        return len(self.df[self.df["split"]=="test"])

    @staticmethod
    def get_valid_split_err(split: str):
        if split not in Dataset.splits:
            raise ValueError(f"misconfiguration error, dataset :split is invalid, expected one of {Dataset.splits}, got {split}") 
        return split

class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, dataset_constructor: Dataset, dataset_config: DatasetConfig, dataloader_config: DataLoaderConfig) -> None:
        super().__init__()
        self.dataset_constructor = dataset_constructor
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
    
    def setup(self, stage):
        _valid_stages = ("fit", "validate", "test", "predict")
        if stage not in _valid_stages:
            raise ValueError(f":stage must be one of {_valid_stages}, got {stage}")
        if stage in ("fit", "validate"):
            self.val_dataset: Dataset = self.dataset_constructor("val", self.dataset_config) 
            if stage == "fit":
                self.train_dataset: Dataset = self.dataset_constructor("train", self.dataset_config) 
        if stage == "test":
            self.test_dataset: Dataset = self.dataset_constructor("test", self.dataset_config) 

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, shuffle = True, **self.dataloader_config.params)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, shuffle = False, **self.dataloader_config.params)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, shuffle = False, **self.dataloader_config.params)

class StreamingDatasetDataModule(LightningDataModule):
    def __init__(self, dataset_constructor: Dataset, dataset_config: DatasetConfig, dataloader_config: DataLoaderConfig) -> None:
        super().__init__()
        self.dataset_constructor = dataset_constructor
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config

    def setup(self, stage):
        _valid_stages = ("fit", "validate", "test", "predict")
        if stage not in _valid_stages:
            raise ValueError(f":stage must be one of {_valid_stages}, got {stage}")
        if stage in ("fit", "validate"):
            self.val_dataset: StreamingDataset = self.dataset_constructor("val", self.dataset_config) 
            if stage == "fit":
                self.train_dataset: StreamingDataset = self.dataset_constructor("train", self.dataset_config) 
        if stage == "test":
            self.test_dataset: StreamingDataset = self.dataset_constructor("test", self.dataset_config) 

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return StreamingDataLoader(self.train_dataset, shuffle = True, **self.dataloader_config.params)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return StreamingDataLoader(self.val_dataset, shuffle = False, **self.dataloader_config.params)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return StreamingDataLoader(self.test_dataset, shuffle = False, **self.dataloader_config.params)

class Compose(Transform):
    """applies a sequence of transforms while skipping over pixel(color) transforms, as they can mess up segmentation masks"""

    def __init__(self, transforms: Sequence[Transform]):
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of transforms")
        elif len(transforms) < 2:
            raise ValueError("Pass at least two transforms")
        self.transforms = transforms

    def forward(self, inputs: tuple[Any, Any]) -> Any:
        for transform in self.transforms:
            if self.is_color_transform(transform):
                continue
            outputs = transform(*inputs)
            inputs = outputs
        return outputs

    def extra_repr(self) -> str:
        format_string = []
        for transform in self.transforms:
            if self.is_color_transform(transform):
                format_string.append(f"   *{transform}")
            else:
                format_string.append(f"    {transform}")
        return "\n".join(format_string)
    
    def is_color_transform(self, transform) -> bool:
        "_color" in str(type(transform))

def _get_stratified_split_df(df: pd.DataFrame, random_seed: int, test_frac: float, val_frac: float, split_on: str) -> pd.DataFrame:
    """returns df split into train-val-test, by stratified(proportionate) sampling, based on
    :split_col and :split_config 
    s.t. num_eval_samples[i] = eval_split * num_samples[i], i -> {0, ..., num_classes-1}.

    Parameters
    - 
    :df -> dataframe to split, must contain string typed column named "label_str"
    :random_seed -> to use for deterministic sampling from dataframe
    :test_frac -> proportion of samples per class for testing 
    :val_frac -> proportion of samples per class for validation
    
    Returns
    -
    pd.DataFrame 

    Raises
    -
    SchemaError
    """ 
    df = pa.DataFrameSchema({split_on: pa.Column(str)})(df)
    assert isinstance(test_frac, float), f"config error (invalid type), expected :test_frac to be of type float, got {type(test_frac)}"
    assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
    assert test_frac + val_frac > 0 and test_frac + val_frac < 1, f"config error (invalid value), expected 0 < :test_frac + :val_frac < 1, got :test_frac={test_frac} and :val_frac={val_frac}"

    test = (df
            .groupby(split_on, group_keys=False)
            .apply(func = lambda x: x.sample(
                    frac = test_frac, random_state = random_seed, axis = 0),
                    include_groups = False
                )
            .assign(split = "test")) 
            
    val = (df
            .drop(test.index, axis = 0)
            .groupby(split_on, group_keys=False)
            .apply(func = lambda x: x.sample( # type: ignore
                    frac = val_frac / (1-test_frac), random_state = random_seed, axis = 0),
                    include_groups = False
                )
            .assign(split = "val"))

    train = (df
            .drop(test.index, axis = 0)
            .drop(val.index, axis = 0)
            .assign(split = "train"))

    # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :df
    return pd.concat([train, val, test]).sort_index()

def _get_random_split_df(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    return pd.DataFrame()

def _get_equal_split_df(df: pd.DataFrame, random_seed: int, test_sample: int, val_sample: int) -> pd.DataFrame:
    return pd.DataFrame()

def _get_imagefolder_split_df(df: pd.DataFrame) -> pd.DataFrame:
    """returns df by assiging splits based on top level parent dir names, assuming {split}/{class_dir}/{sample_image}
    format in df["image_path"]"""
    return df.assign(split = lambda df: df["image_path"].apply(lambda x: x.split('/')[0]))

def _get_imagefolder_notest_split_df(df: pd.DataFrame, random_seed:int, val_frac: float, split_on: str) -> pd.DataFrame:
    """returns df by where samples in the val/ dir are assigned split = test, and stratified samples from the train/ dir are assigned
     split = val based on :val_frac; rest are assigned split = train"""

    df = pa.DataFrameSchema({split_on: pa.Column(str)})(df)
    assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
    assert val_frac > 0 and val_frac < 1.0, f"config error (invalid value), expected 0 < :val_frac < 1, got :val_frac={val_frac}"

    # list all samples inside val/ and assign split = test
    test_filter = df["image_path"].apply(lambda x: "val" in str(x))
    if len(test_filter) == 0:
        raise KeyError("couldn't find samples inside 'val/' dir in the df")
    test = (
        df[test_filter]
        .assign(split = "test")
    )

    # sample from inside train/ and assign split = val 
    val = (
        df.drop(test.index, axis = 0)
        .groupby("label_str", group_keys=False)
        .apply(lambda x: x.sample(frac = val_frac, random_state = random_seed, axis = 0), include_groups = True)
        .assign(split = "val")
    )

    # assign split = train to remaining samples
    train = (
        df
        .drop(test.index, axis = 0)
        .drop(val.index, axis = 0)
        .assign(split = "train")
    )
    return pd.concat([train, val, test]).sort_index()