from numpy.typing import NDArray
from typing import Any, Optional, Literal, Callable, Sequence
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from functools import cache
from itertools import product
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

    def _get_tabular_sampler(self, name: str):  # noqa: F821
        if name == "stratified":
            return self.stratified_tabular_sampler
        elif name == "equal":
            return self.equals_tabular_sampler
        elif name == "imagefolder":
            return self.imagefolder_tabular_sampler
        elif name == "imagenet":
            return self.imagenet_tabular_sampler
        else:
            raise AssertionError(f"not implemented error, {name} has not been implemented/registered yet")
    
    def _get_spatial_sampler(self, name: str):
        if name == "sliding_window":
            return self.sliding_window_spatial_sampler
        else:
            raise AssertionError(f"not implemented error, {name} has not been implemented/registered yet")

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
    def stratified_tabular_sampler(index_df: pd.DataFrame, test_frac: float, val_frac: float, split_on: str, random_seed: int) -> pd.DataFrame:
        """
        returns df split into train-val-test, by stratified(proportionate) sampling, based on :split_col (class label) such that for each (i^th)
        class, i ∈ {0, ..., num_classes-1} and eval ∈ {val, test}, num_:eval_samples[i] = :eval_frac * num_samples[i]

        Parameters:
            :df -> table to resample
            :random_seed -> to use for deterministic sampling from dataframe
            :test_frac -> proportion of samples per class for testing 
            :val_frac -> proportion of samples per class for validation
            :split_on -> column used to group by, must be present in :df
        
        """ 
        assert split_on in index_df.columns 
        assert isinstance(test_frac, float), f"config error (invalid type), expected :test_frac to be of type float, got {type(test_frac)}"
        assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
        assert test_frac + val_frac > 0 and test_frac + val_frac < 1, \
            f"config error (invalid value), expected 0 < :test_frac + :val_frac < 1, got :test_frac={test_frac} and :val_frac={val_frac}"

        test = (
            index_df
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(frac=test_frac, random_state=random_seed, axis=0), include_groups=True)
            .assign(split = "test")
        ) 
        val = (
            index_df
            .drop(test.index, axis = 0)
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(frac=val_frac/(1-test_frac), random_state=random_seed, axis=0), include_groups=True) # type: ignore
            .assign(split="val")
        )

        train = (
            index_df
            .drop(test.index, axis = 0)
            .drop(val.index, axis = 0)
            .assign(split = "train")
        )

        # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :index_df
        return pd.concat([train, val, test]).sort_index()

    @staticmethod
    def equal_tabular_sampler(index_df: pd.DataFrame, num_val_samples: int, num_test_samples: int, split_on: str, random_seed: int) -> pd.DataFrame:
        assert split_on in index_df.columns 
        assert isinstance(num_val_samples, int) and num_val_samples >= 0
        assert isinstance(num_test_samples, int) and num_test_samples >= 0
        assert num_val_samples + num_test_samples < len(index_df)

        test = (
            index_df
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(n = num_test_samples, random_state=random_seed, axis=0), include_groups=True)
            .assign(split = "test")
        )

        val = (
            index_df
            .drop(index=test.index)
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(n = num_val_samples, random_state=random_seed, axis=0), include_groups=True) # type: ignore
            .assign(split="val")
        )

        train = (
            index_df
            .drop(index=test.index)
            .drop(index=val.index)
            .assign(split = "train")
        )

        # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :index_df
        return pd.concat([train, val, test]).sort_index()

    @staticmethod
    def imagenet_tabular_sampler(index_df: pd.DataFrame, val_frac: int, split_on: str, random_seed: int) -> pd.DataFrame:
        """
        returns df where split=test is assigned to samples with 'val' in their paths, and split=val sampled (:val_frac) proportionally from the 
        remaining samples using the column named :split_on as the class label.
        """
        assert split_on in index_df.columns
        assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
        assert val_frac >= 0.0 and val_frac < 1.0, f"config error (invalid value), expected 0 <= :val_frac < 1, got :val_frac={val_frac}"

        # list all samples inside val/ and assign split = test
        test = index_df.loc[lambda df: df["image_path"].apply(lambda x: "val" in str(x))].assign(split = "test")

        # sample from inside train/ and assign split = val 
        val = (
            index_df.drop(index=test.index)
            .groupby(split_on, group_keys=False)
            .apply(lambda x: x.sample(frac = val_frac, random_state = random_seed, axis = 0), include_groups = True)
            .assign(split = "val")
        )

        # assign split = train to remaining samples
        train = (
            index_df
            .drop(test.index, axis = 0)
            .drop(val.index, axis = 0)
            .assign(split = "train")
        )
        return pd.concat([train, val, test]).sort_index()

    @staticmethod
    def imagefolder_tabular_sampler(
            index_df: pd.DataFrame, 
            random_seed: int, 
            val_split: Optional[int | float] = None,
            split_on: Optional[str] = None 
        ) -> pd.DataFrame:
        """
        returns df by assiging split based on the grand-parent dir, i.e., assuming a ../{split}/{class}/{image} format in :index_df["image_path"].
        if the val/ dir is not found, :val_split samples from the train/ dir are assigned split=val using the :split_on column. 
        if the test/ dir is not found, samples from the val/ dir are assigned split=test, and :val_split samples from the train/ dir are assigned 
        split=val using the :split_on column. raises Assertion error if both test/ and val/ are missing.

        Parameters
        -
        :index_df -> table to resample.
        :random_seed -> for deterministic sampling.
        :val_frac -> if integer, specifies the number of samples from train/ to resample as val/. if float, specifies proportion per 
        class of the samples in train/ to resample as val/ .
        :split_on -> column in :df containing the parent class names, used to group by, defaults to class_dir if not specified

        """
        def get_val(train_df: pd.DataFrame) -> pd.DataFrame:
            val_df = train_df.groupby(split_on, group_keys=False)
            if isinstance(val_split, int):
                val_df = val_df.apply(lambda x: x.sample(n = val_split, random_state=random_seed, axis=0), include_groups=True) # type: ignore
            elif isinstance(val_split, float):
                val_df = val_df.apply(lambda x: x.sample(frac = val_split, random_state=random_seed, axis=0), include_groups=True) # type: ignore
            val_df = val_df.assign(split="val")
            return val_df

        index_df["split"] = index_df["image_path"].apply(lambda x: str(x).split('/')[-3])

        splits = set(index_df["split"].unique())
        if "test" not in splits or "val" not in splits:
            assert split_on in index_df.columns
            assert val_split is not None and isinstance(val_split, (int, float))

            if "test" not in splits:
                assert "train" in splits and "val" in splits
                test = index_df[index_df["split"] == "val"]
            elif "val" not in splits:
                assert "train" in splits and "test" in splits
                test = index_df[index_df["split"] == "test"]
            train = index_df[index_df["split"] == "train"]
            val = get_val(train)
            train = train.drop(index=val.index)
            return pd.concat([train, val, test]).sort_index()
        return index_df

    @staticmethod
    def sliding_window_spatial_sampler(
            index_df: pd.DataFrame,
            spatial_df: pd.DataFrame,
            tile_size: int | Sequence[int],
            tile_stride: int | Sequence[int],
            **kwargs
        ) -> pd.DataFrame:
        """
        returns df with tile_x_min, x_max, y_min and y_max , indicating the pixel coordinates of the top left and bottom right corners of the image 
        tile respectively. these are calculated by sliding a window of :tile_size over the image with :tile_stride. this sampler expects the image 
        bounds to be specified using 'image_width' and 'image_height' columns in the spatial_df.

        Parameters
        -
        :index_df > df to resample\n
        :spatial_df > df with spatial info, specifially image_height and image_width\n 
        :tile_size > size of tile in pixels used to calculate tile bounds. int is converted to tuple[int, int]\n
        :tile_stride > stride of sliding window in pixels used to calculate tile bounds. int is converted to tuple[int, int]\n
        """
        
        @cache
        def get_min(length: int, stride: int) -> NDArray:
            return np.arange(start=0, stop=length, step=stride, dtype=np.uint16)
        
        assert index_df.index.equals(spatial_df.index)
        assert "image_height" in spatial_df.columns
        assert "image_width" in spatial_df.columns

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if isinstance(tile_stride, int):
            tile_stride = (tile_stride, tile_stride)

        assert isinstance(tile_size, Sequence) and len(tile_size) == 2
        assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

        top_left = {k:list() for k in ("idx", "y_min", "x_min")}
        for idx, row in spatial_df.iterrows():
            y_mins = get_min(row["image_height"], tile_stride[0])
            x_mins = get_min(row["image_width"], tile_stride[1])
            for y_min, x_min in product(y_mins, x_mins):
                top_left["idx"].append(idx)
                top_left["y_min"].append(y_min)
                top_left["x_min"].append(x_min)
            
        df = pd.DataFrame(top_left).set_index("idx").merge(index_df, how = "left", left_index=True, right_index=True)
        df["y_max"] = df["y_min"] + tile_size[0]
        df["x_max"] = df["x_min"] + tile_size[1]
        df = df[index_df.columns.to_list() + ["y_min", "y_max", "x_min", "x_max"]]
        return df

    @classmethod
    def band_combination_spectral_sampler(
            index_df: pd.DataFrame,
            spectral_df: pd.DataFrame,
            bands: Sequence[int] 
        ) -> pd.DataFrame:
        """
        returns df with the chosen :bands subset from the image. this fn expects the number of channels in the image to be specified by a column 
        named num_channels in the spectral_df

        Parameters
        -
        :index_df -> table to be resampled
        :spectral_df -> table containing spectral information
        :bands -> bands to be subset from the multispectral image 
        """
        ...

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
    task: str
    subtask: str 
    split: str
    storage: str
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

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"], config: Optional[DatasetConfig]) -> None: # type: ignore  # noqa: F821
        # Check init args
        logger.info(f"attempting to init {self.name}_{self.storage}_{self.task}_{self.subtask}")
        assert hasattr(self, "name") and isinstance(self.name, str)
        assert hasattr(self, "task") and self.task in self.valid_tasks 
        assert hasattr(self, "subtask") and self.subtask in getattr(self, f"valid_{self.task}_subtasks")
        assert hasattr(self, "storage") and self.storage in self.valid_storage_formats
        assert hasattr(self, "class_names") and isinstance(self.class_names, tuple)
        assert hasattr(self, "num_classes") and isinstance(self.num_classes, int) # and self.num_classes == len(self.class_names)

        # Verify files exist on 
        assert hasattr(self, "root")
        if self.storage in ("imagefolder", "litdata"):
            fs.get_valid_dir_err(self.root, empty_ok = False)# self.root.is_dir()
        elif self.storage == "hdf5" :
            assert self.root.is_file() 

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
                    df = sampler(index_df = df, **params)
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