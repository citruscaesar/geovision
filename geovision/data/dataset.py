from typing import Literal, Optional, Any

from pathlib import Path
from pandas import DataFrame
from abc import ABC, abstractmethod 
from .config import DatasetConfig

class Dataset(ABC):
    valid_splits = ("train", "val", "test", "trainval", "all")

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
    def df(self) -> DataFrame:
        pass

    @property
    @abstractmethod
    def split_df(self) -> DataFrame:
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