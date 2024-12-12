from typing import Any, Optional, Literal, Callable
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import logging
import pandas as pd

from pathlib import Path
from pandera import DataFrameSchema
from lightning import LightningDataModule
from torch.utils.data import default_collate
from torchvision.transforms.v2 import Transform, Identity, CutMix, MixUp

from torch.utils.data import DataLoader
from geovision.io.local import FileSystemIO as fs 
from litdata import StreamingDataLoader, StreamingDataset

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
            assert self.df_path.suffix in (".csv", ".h5", ".parquet"), f"expected :df_path as one of .csv, .h5 or .parquet, got {self.df_path.suffix}"
            if self.df_path.suffix == ".csv":
                self.df = pd.read_csv(self.df_path)
            elif self.df_path.suffix == ".h5":
                self.df = pd.read_hdf(self.df_path, mode = "r")
            elif self.df_path.suffix == ".parquet":
                self.df = pd.read_parquet(self.df_path)
        else:
            self.df = None 
            self.df_path = None

        for sampler in ("tabular", "spatial", "spectral", "temporal"):
            setattr(self, f"{sampler}_sampler_name", locals().get(f"{sampler}_sampler_name"))
            sampler_name = getattr(self, f"{sampler}_sampler_name")
            if sampler_name is not None:
                setattr(self, f"{sampler}_sampler", getattr(self, f"_get_{sampler}_sampler")(sampler_name))
                setattr(self, f"{sampler}_sampler_params", (locals().get(f"{sampler}_sampler_params") or dict()) | {"random_seed": self.random_seed})
            else:
                setattr(self, f"{sampler}_sampler", None)
                setattr(self, f"{sampler}_sampler_params", None)

        self.image_pre = image_pre or Identity()
        self.target_pre = target_pre or Identity()
        self.train_aug = train_aug or Identity()
        self.eval_aug = eval_aug or Identity()

        # self.tabular_sampler_name = tabular_sampler_name
        # if self.tabular_sampler_name is not None:
            # self.tabular_sampler = self._get_tabular_sampler(tabular_sampler_name)
            # self.tabular_sampler_params = ( tabular_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        # else:
            # self.tabular_sampler = None
            # self.tabular_sampler_params = None

        # self.spatial_sampler_name = spatial_sampler_name
        # if self.spatial_sampler_name is not None:
            # self.spatial_sampler = self._get_spatial_sampler(spatial_sampler_name)
            # self.spatial_sampler_params = ( spatial_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        # else:
            # self.spatial_sampler = None
            # self.spatial_sampler_params = None

        # self.spectral_sampler_name = spectral_sampler_name
        # if self.spectral_sampler_name is not None:
            # self.spectral_sampler = self._get_spectral_sampler(spectral_sampler_name)
            # self.spectral_sampler_params = ( spectral_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        # else:
            # self.spectral_sampler = None
            # self.spectral_sampler_params = None

        # self.temporal_sampler_name = temporal_sampler_name
        # if self.temporal_sampler_name is not None:
            # self.temporal_sampler = self._get_temporal_sampler(temporal_sampler_name)
            # self.temporal_sampler_params = ( temporal_sampler_params or dict() ) | {"random_seed" : self.random_seed}
        # else:
            # self.temporal_sampler = None
            # self.temporal_sampler_params = None

    def _get_tabular_sampler(self, name: str):  # noqa: F821
        if name == "stratified":
            return self._get_stratified_split_df
        elif name == "equal":
            return self._get_equal_split_df
        elif name == "random":
            return self._get_random_split_df
        elif name ==  "imagefolder_notest":
            return self._get_imagefolder_notest_split_df
        else:
            raise AssertionError(f"not implemented error, {name} has not been implemented/registered yet")
    
    def _get_spatial_sampler(self, name: str):
        return None

    def _get_spectral_sampler(self, name: str):
        return None

    def _get_temporal_sampler(self, name: str):
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

    @staticmethod
    def _get_stratified_split_df(df: pd.DataFrame, random_seed: int, test_frac: float, val_frac: float, split_on: str) -> pd.DataFrame:
        """
        returns df split into train-val-test, by stratified(proportionate) sampling, based on :split_col and :split_config s.t. 
        num_eval_samples[i] = eval_split * num_samples[i], where i -> {0, ..., num_classes-1}.

        Parameters:
            :df -> table to resample
            :random_seed -> to use for deterministic sampling from dataframe
            :test_frac -> proportion of samples per class for testing 
            :val_frac -> proportion of samples per class for validation
            :split_on -> column used to group by, must be present in :df
        
        """ 
        assert split_on in df.columns 
        assert isinstance(test_frac, float), f"config error (invalid type), expected :test_frac to be of type float, got {type(test_frac)}"
        assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
        assert test_frac + val_frac > 0 and test_frac + val_frac < 1, \
            f"config error (invalid value), expected 0 < :test_frac + :val_frac < 1, got :test_frac={test_frac} and :val_frac={val_frac}"

        test = (
            df
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(frac=test_frac, random_state=random_seed, axis=0), include_groups=True)
            .assign(split = "test")
        ) 
        val = (
            df
            .drop(test.index, axis = 0)
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(frac=val_frac/(1-test_frac), random_state=random_seed, axis=0), include_groups=True) # type: ignore
            .assign(split="val")
        )

        train = (df
                .drop(test.index, axis = 0)
                .drop(val.index, axis = 0)
                .assign(split = "train"))

        # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :df
        return pd.concat([train, val, test]).sort_index()

    @staticmethod
    def _get_random_split_df(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _get_equal_split_df(df: pd.DataFrame, random_seed: int, test_sample: int, val_sample: int) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _get_imagefolder_split_df(df: pd.DataFrame) -> pd.DataFrame:
        """returns df by assiging splits based on top level parent dir names, assuming {split}/{class_dir}/{sample_image}
        format in df["image_path"]"""
        return df.assign(split = lambda df: df["image_path"].apply(lambda x: x.split('/')[0]))

    @staticmethod
    def _get_imagefolder_notest_split_df(df: pd.DataFrame, random_seed:int, val_frac: float, split_on: str) -> pd.DataFrame:
        """
        returns df where samples in the val/ directory are assigned split = test, and samples from the train/ dir are assigned split = train with a
        stratified subset being assigned split = val based on :val_frac

        Parameters
        :df -> table to resample.
        :random_seed -> for deterministic sampling.
        :val_frac -> proportion of the images in train/ split per class to resample as val.
        :split_on -> column in :df containing the parent class names, used to group by

        """

        assert split_on in df.columns
        assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
        assert val_frac > 0 and val_frac < 1.0, f"config error (invalid value), expected 0 < :val_frac < 1, got :val_frac={val_frac}"

        # list all samples inside val/ and assign split = test
        test_filter = df["image_path"].apply(lambda x: "val" in str(x))
        if len(test_filter) == 0:
            raise KeyError("couldn't find samples inside 'val/' dir in the df")
        test = df[test_filter].assign(split = "test")

        # sample from inside train/ and assign split = val 
        val = (
            df.drop(test.index, axis = 0)
            .groupby(split_on, group_keys=False)
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
            elif batch_transform_name == "mixup":
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

class Dataset:
    name: str
    task: Literal["train", "val", "test", "trainvaltest", "all"]
    subtask: str 
    storage: Literal["imagefolder", "hdf5", "litdata", "memory"]
    class_names: tuple[str, ...]
    num_classes: int
    root: Path
    config: DatasetConfig  # type: ignore # noqa: F821
    schema: DataFrameSchema
    loader: Callable[..., pd.DataFrame]

    valid_splits = ("train", "val", "test", "trainvaltest", "all")
    valid_tasks = ("classification", "segmentation", "super_resolution", "detection", "unsupervised")
    valid_storage_formats = ("imagefolder", "hdf5", "litdata", "memory")
    valid_classification_subtasks = ("multiclass", "multilabel")
    valid_segmentation_subtasks = ("semantic", "instance", "panoptic")
    valid_super_resolution_subtasks = ()
    valid_detection_subtasks = ()

    def __init__(self, split: str, config: Optional[DatasetConfig]) -> None: # type: ignore  # noqa: F821
        logger.info(f"attempting to init {self.name}_{self.storage}_{self.task}_{self.subtask}")
        

        assert hasattr(self, "name") and isinstance(self.name, str)
        assert hasattr(self, "task") and self.task in self.valid_tasks 
        assert hasattr(self, "subtask") and self.subtask in getattr(self, f"valid_{self.task}_subtasks")
        assert hasattr(self, "storage") and self.storage in self.valid_storage_formats

        assert hasattr(self, "class_names") and isinstance(self.class_names, tuple)
        assert hasattr(self, "num_classes") and isinstance(self.num_classes, int) # and self.num_classes == len(self.class_names)

        assert hasattr(self, "root")
        assert hasattr(self, "config") and isinstance(self.config, DatasetConfig) 
        assert hasattr(self, "schema") and isinstance(self.schema, DataFrameSchema) 
        assert hasattr(self, "loader") and callable(self.loader) 

        assert split in self.valid_splits, f"value error (invalid), expected :split to be one of {self.valid_splits}, got {split}"
        self.split = split

        if config is not None:
            self.config = config

        if self.config.df is None:
            df = self.loader("index", self.storage, self.name) 
            for sampler_name in ("tabular", "spatial", "spectral", "temporal"):
                sampler: Callable = getattr(self.config, f"{sampler_name}_sampler")
                params: dict = getattr(self.config, f"{sampler_name}_sampler_params")
                if sampler is not None:
                    logger.info(f"found {sampler_name} sampling, using fn: {sampler} and params: {params}")
                    if sampler_name != "tabular":
                        params[f"{sampler_name}_df"] = self.loader(sampler_name, self.storage, self.name)
                    df = sampler(df = df, **params)
                    logger.info(f"successfully applied {sampler_name} sampling")
                else:
                    logger.info(f"did not find {sampler_name} sampler, skipping")
        else:
            df = self.config.df
        self.index_df = self.schema(df)

        logger.info(
            f"""
                successfully init {self.name}_{self.storage}_{self.task}_{self.subtask} with augmentations image_pre = {self.config.image_pre}\n
                target_pre = {self.config.target_pre}\n train_aug = {self.config.train_aug}\neval_aug = {self.config.eval_aug}
            """
        )
       
    def __repr__(self) -> str:
        return '\n'.join([
            f"{self.name} dataset for {self.task}_{self.subtask}",
            f"local {self.storage} @ [{self.root}] ",
            f"with {len(self.class_names)} classes and {len(self)} images under the '{self.split}' split",
        ])

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> tuple:
        raise NotImplementedError()

    @property
    def num_total_samples(self) -> int: 
        return len(self.index_df)

    @property
    def num_train_samples(self) -> int: 
        return len(self.index_df[self.index_df["split"]=="train"])

    @property
    def num_val_samples(self) -> int: 
        return len(self.index_df[self.index_df["split"]=="val"])

    @property
    def num_test_samples(self) -> int: 
        return len(self.index_df[self.index_df["split"]=="test"])

    def get_df(self, prefix_root_to_paths: bool) -> pd.DataFrame:
        """returns a (re-indexed) subset of :df with rows containing :split columns, and performs a schema check at the end. raises value error if :split is invalid,
        schema error if df does not have a column named split or split_df fails schema check"""

        assert "split" in self.index_df.columns, "schema error, :index_df does not 'split' column, i.e. train-val-test splits have not been assigned"

        df = self.index_df.assign(df_idx = lambda df: df.index).reset_index(drop = True)

        if self.split not in ("all", "trainvaltest"):
            df = df[df.split == self.split]

        if prefix_root_to_paths:
            if "image_path" in df.columns:
                #df["image_path"] = df["image_path"].apply(lambda x: '/'.join([self.root, *x.split('/')]))
                df["image_path"] = df["image_path"].apply(lambda x: str(Path(self.root, x)))
            if "mask_path" in df.columns:
                #df["mask_path"] = df["mask_path"].apply(lambda x: '/'.join([self.root, *x.split('/')]))
                df["mask_path"] = df["mask_path"].apply(lambda x: str(Path(self.root, x)))
        return df

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