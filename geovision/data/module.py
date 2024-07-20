from typing import Any

from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from .dataset import Dataset
from geovision.config.config import ExperimentConfig

class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.dataloader_kwargs = {
            "batch_size": config.dataloader_params.batch_size // config.dataloader_params.gradient_accumulation,
            "num_workers": config.dataloader_params.num_workers,
            "persistent_workers": config.dataloader_params.persistent_workers,
            "pin_memory": config.dataloader_params.pin_memory
        } 
    
    def setup(self, stage):
        _valid_stages = ("fit", "validate", "test", "predict")
        if stage not in _valid_stages:
            raise ValueError(f":stage must be one of {_valid_stages}, got {stage}")
        if stage in ("fit", "validate"):
            self.val_dataset: Dataset = self.config.get_dataset("val") 
            if stage == "fit":
                self.train_dataset: Dataset = self.config.get_dataset("train") 
        if stage == "test":
            self.test_dataset: Dataset = self.config.get_dataset("test") 

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, shuffle = True, **self.dataloader_kwargs)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, shuffle = False, **self.dataloader_kwargs)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, shuffle = False, **self.dataloader_kwargs)