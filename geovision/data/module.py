from typing import Optional, Callable 

from pandas import DataFrame
from pathlib import Path
from torch.utils.data import DataLoader
# from torchvision.transforms.v2 import Transform # type: ignore
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from geovision.data import get_dataset_from_key 
from geovision.config import ExperimentConfig
from .dataset import Dataset

class ImageDatasetDataModule(LightningDataModule):
    def __init__(
            self,
            config: ExperimentConfig,
            dataset: Optional[Callable] = None,
            dataset_root: Optional[Path | str] = None,
            dataset_df: Optional[DataFrame | Path | str] = None,
    ) -> None:

        self.config = config
        self.num_workers = config.dataloader_params.num_workers
        self.batch_size = (
            config.dataloader_params.batch_size // config.dataloader_params.gradient_accumulation
        )

        self.dataset: Callable 
        if dataset is None:
            self.dataset = get_dataset_from_key(config.dataset_name)
        else:
            self.dataset = dataset

        self.root: Path
        if dataset_root is None:
            self.root = config.dataset_root
        else:
            self.root = Path(dataset_root).expanduser()

        self.df: DataFrame | Path | None
        if dataset_df is None:
            self.df = self.config.dataset_df
        elif isinstance(dataset_df, str):
            self.df = Path(dataset_df).expanduser()
        else:
            self.df = dataset_df
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage):
        _valid_stages = ("fit", "validate", "test", "predict")
        if stage not in _valid_stages:
            raise ValueError(f":stage must be one of {_valid_stages}, got {stage}")
        if stage in ("fit", "validate"):
            self.val_dataset = self._get_val_dataset()
            if stage == "fit":
                self.train_dataset = self._get_train_dataset()
        if stage == "test":
            self.test_dataset = self._get_test_dataset()
    
    def _get_dataset_kwargs(self) -> dict:
        return {
            "root": self.root,
            "df": self.df,
            "df_config": self.config.dataframe_params,
            "transform": self.config.transform
        }
    
    def _get_train_dataset(self) -> Dataset:
        return self.dataset(
            split = "train",
            **self._get_dataset_kwargs()
        )
    
    def _get_val_dataset(self) -> Dataset:
        return self.dataset(
            split = "val",
            **self._get_dataset_kwargs()
        )

    def _get_test_dataset(self) -> Dataset:
        return self.dataset(
            split = "test",
            **self._get_dataset_kwargs()
        )

    def _get_dataloader_kwargs(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "pin_memory": True,
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = self.train_dataset,  
            shuffle = True,
            **self._get_dataloader_kwargs()
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = self.val_dataset,
            shuffle = False,
            **self._get_dataloader_kwargs()
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = self.test_dataset,
            shuffle = False,
            **self._get_dataloader_kwargs()
        )
