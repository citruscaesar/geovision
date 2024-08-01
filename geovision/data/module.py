from .dataset import Dataset
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
    
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
        return DataLoader(self.train_dataset, shuffle = True, **self.config.get_dataloader_params())
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, shuffle = False, **self.config.get_dataloader_params())

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, shuffle = False, **self.config.get_dataloader_params())