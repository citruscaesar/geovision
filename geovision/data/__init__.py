from typing import Optional, Literal, Callable
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import logging
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from pandera.pandas import DataFrameSchema
from geovision.io.local import FileSystemIO as fs 

from geovision.experiment.config import Builder, DatasetBuilder, DatasetConfigParams, DataLoaderConfigParams

type Split = Literal["train", "val", "test", "trainvaltest", "all"]

logger = logging.getLogger(__name__)

class Dataset:
    name: str
    task: str
    subtask: str 
    split: Split 
    storage: str
    class_names: tuple[str, ...]
    num_classes: int
    root: Path
    schema: DataFrameSchema
    loader: Callable[..., pd.DataFrame]
    metadata_group_prefix: Optional[str] # prefix to hdf5 keys

    valid_splits = ("train", "val", "test", "trainvaltest", "all")
    valid_tasks = ("classification", "segmentation", "super_resolution", "detection", "unsupervised")
    valid_storage_formats = ("imagefolder", "hdf5", "litdata", "memory")
    valid_classification_subtasks = ("multiclass", "multilabel")
    valid_segmentation_subtasks = ("semantic", "instance", "panoptic")
    valid_super_resolution_subtasks = ()
    valid_detection_subtasks = ()
    valid_unsupervised_subtasks = ("pretraining")

    def __init__(self, split: Split, config: DatasetConfigParams) -> None: # type: ignore  # noqa: F821
        self.__check_init_args()
        self.__check_storage()
        self.__check_split(split)
        self.__check_config(config)

        # if a df is loaded from the config.yaml, use that. Else use the loader function to get index_df from the metadata, and resample as specified
        if self.config.df is None:
            df = self.loader(self.metadata_group_prefix + "index", self.storage, self.name) 
            df = self.__resample_df(df)
        else:
            df = self.config.df
        self.index_df = self.schema(df)

        self.__log_success()
      
    def __repr__(self) -> str:
        return '\n'.join([
            f"{self.name} dataset for {self.subtask} {self.task}",
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
        """returns a (re-indexed) subset of :df with rows containing rows belonging to current split, and performs a schema check at the end. 
        raises assertion (schema) error if df does not have a column named split or split_df fails schema check"""

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

    def __check_init_args(self):
        logger.info(f"attempting to init {self.name}_{self.storage}_{self.task}_{self.subtask}")
        assert hasattr(self, "name") and isinstance(self.name, str)
        assert hasattr(self, "task") and self.task in self.valid_tasks 
        assert hasattr(self, "subtask") and self.subtask in getattr(self, f"valid_{self.task}_subtasks")
        assert hasattr(self, "storage") and self.storage in self.valid_storage_formats
        assert hasattr(self, "class_names") and isinstance(self.class_names, tuple)
        assert hasattr(self, "num_classes") and isinstance(self.num_classes, int) # and self.num_classes == len(self.class_names)
        assert hasattr(self, "schema") and isinstance(self.schema, DataFrameSchema) 
        assert hasattr(self, "loader") and callable(self.loader) 

        assert hasattr(self, "metadata_group_prefix")
        # if none
        if isinstance(self.metadata_group_prefix, str):
            assert self.metadata_group_prefix.endswith('/')
        elif self.metadata_group_prefix is None: 
            self.metadata_group_prefix = str()
        assert isinstance(self.metadata_group_prefix, str)

    def __check_storage(self):
        assert hasattr(self, "root")
        if self.storage in ("imagefolder", "litdata"):
            fs.get_valid_dir_err(self.root, empty_ok = False) # self.root.is_dir()
        elif self.storage == "hdf5" :
            assert self.root.is_file() 
    
    def __check_split(self, split: Optional[Split]):
        split = split or "all" # fallback value 
        assert split in self.valid_splits, f"value error (invalid), expected :split to be one of {self.valid_splits}, got {split}"
        self.split = split
    
    def __check_config(self, config: DatasetConfigParams):
        # assert isinstance(config, DatasetConfigParams), f"type error, expected dataset.config to be of type DatasetConfigParams, got {type(config)}"
        self.config = config
    
    def __resample_df(self, df: pd.DataFrame) -> pd.DataFrame:
        for sampler_name in ("tabular", "spatial", "spectral", "temporal"):

            sampler: Builder = getattr(self.config, f"{sampler_name}_sampler")

            if sampler is not None:
                #params: dict = getattr(self.config, f"{sampler_name}_sampler_params")
                logger.info(f"found {sampler_name} sampling, using fn: {sampler.constructor} and params: {sampler.kwargs}")

                sampler.kwargs["index_df"] = df
                if sampler_name != "tabular":
                    # if any associated metadata is needed but hasn't been loaded yet
                    if sampler.kwargs.get(f"{sampler_name}_df") is None:
                        try:
                            sampler.kwargs[f"{sampler_name}_df"] = self.loader(self.metadata_group_prefix + sampler_name, self.storage, self.name)
                        except Exception:
                            sampler.kwargs[f"{sampler_name}_df"] = None 
                
                # apply resampling, .build() is equivalent to sampler.constructor(**sampler.kwargs), which returns a DataFrame
                df: pd.DataFrame = sampler.build() #df = sampler(index_df = df, **sampler_)
                logger.info(f"successfully applied {sampler_name} sampling")
            else:
                logger.info(f"did not find {sampler_name} sampler, skipping")

        return df

    def __log_success(self):
        logger.info(
            f"""
                successfully init {self.name}_{self.storage}_{self.task}_{self.subtask} with augmentations image_pre = {self.config.image_pre}\n
                target_pre = {self.config.target_pre}\n train_aug = {self.config.train_aug}\neval_aug = {self.config.eval_aug}
            """
        )
 
class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, dataset_builder: DatasetBuilder, dataloader_config: DataLoaderConfigParams) -> None:
        super().__init__()
        self.dataset_builder = dataset_builder
        self.dataloader_config = dataloader_config
           
    def setup(self, stage):
        _valid_stages = ("fit", "validate", "test", "predict")
        assert stage in _valid_stages, f"value error, expected DataModule.stage to be one of {_valid_stages}, got {stage}"

        if stage in ("fit", "validate"):
            self.val_dataset: Dataset = self.dataset_builder.build(split = "val") 
            if stage == "fit":
                self.train_dataset: Dataset = self.dataset_builder.build(split = "train") 
        if stage == "test":
            self.test_dataset: Dataset = self.dataset_builder.build(split = "test") 

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, shuffle = True, **self.dataloader_config.kwargs)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, shuffle = False, **self.dataloader_config.kwargs)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, shuffle = False, **self.dataloader_config.kwargs)