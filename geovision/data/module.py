from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from geovision.data import get_dataset_from_key 
from geovision.config import ExperimentConfig
from .dataset import Dataset

class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        self.dataset = get_dataset_from_key(config.dataset.name)
        self.root = config.dataset.root
        self.df = config.dataset.df

        self.dataset_config = config.dataset
        self.transforms_config = config.transforms
        self.batch_size = config.dataloader.batch_size // config.dataloader.gradient_accumulation
        self.num_workers = config.dataloader.num_workers
    
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
            "config": self.dataset_config,
            "transforms": self.transforms_config
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
