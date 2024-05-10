from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from geovision.config.basemodels import ExperimentConfig

from .dataset import Dataset

class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
    
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
            "root": self.config.dataset_root,
            "df": self.config.dataset_df,
            "config": self.config.dataset_config,
            "transforms": self.config.transforms
        }
    
    def _get_train_dataset(self) -> Dataset:
        return self.config.dataset(
            split = "train",
            **self._get_dataset_kwargs()
        )
    
    def _get_val_dataset(self) -> Dataset:
        return self.config.dataset(
            split = "val",
            **self._get_dataset_kwargs()
        )

    def _get_test_dataset(self) -> Dataset:
        return self.config.dataset(
            split = "test",
            **self._get_dataset_kwargs()
        )

    def _get_dataloader_kwargs(self) -> dict:
        kwargs = self.config.dataloader_config.model_dump()
        kwargs["batch_size"] = kwargs["batch_size"] // kwargs["gradient_accumulation"]
        return kwargs 

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
