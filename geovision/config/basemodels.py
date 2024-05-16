from typing import Any, Callable, Optional

import yaml # type: ignore
import torch
import torchvision
import torchmetrics
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator 
from torchvision.transforms.v2 import Transform

from geovision.io.local import get_valid_file_err
from geovision.data.dataset import Dataset, DatasetConfig, TransformsConfig
from geovision.data.imagenette import (
    ImagenetteImagefolderClassification,
    ImagenetteHDF5Classification
)

class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    gradient_accumulation: int
    persistent_workers: bool
    pin_memory: bool

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True)
    name: str
    dataset: Dataset | str
    dataset_root: Path | str
    dataset_df: Path | str | None
    dataset_params: DatasetConfig | None
    dataloader_params: DataLoaderConfig
    # model: torch.nn.Module | str
    # model_params: dict
    criterion: torch.nn.Module | str
    criterion_params: dict
    optimizer: torch.optim.Optimizer | str
    optimizer_params: dict
    metric: str
    transforms: TransformsConfig | None

    @field_validator("dataset")
    @classmethod
    def get_dataset(cls, name: str) -> Dataset:
        datasets = {
            "imagenette_imagefolder_classification": ImagenetteImagefolderClassification,
            "imagenette_hdf5_classification": ImagenetteHDF5Classification 
        }
        return cls._get_fn_from_table(datasets, name, "dataset") #type: ignore

    @field_validator("dataset_root")
    @classmethod
    def get_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root).resolve().expanduser()
        if not root.exists():
            raise FileNotFoundError(f"dataset root not found on local fs, got {root}")
        return root

    @field_validator("dataset_df")
    @classmethod
    def get_dataset_df(cls, df: str | Path | None) -> Path | None:
        if df is not None:
            return get_valid_file_err(df, valid_extns=(".csv",))
        return None

    # @field_validator("model")
    # @classmethod
    # def get_model(cls, name: str) -> Callable:
        # models = {
            # "resnet18": torchvision.models.resnet18,
            # "resnet34": torchvision.models.resnet34
        # }
        # return cls._get_fn_from_table(models, name, "model")

    @field_validator("criterion")
    @classmethod
    def get_criterion(cls, name: str) -> Callable:
        criterions = {
            "binary_cross_entropy": torch.nn.BCEWithLogitsLoss,
            "cross_entropy": torch.nn.CrossEntropyLoss,
            "mean_squared_error": torch.nn.MSELoss,
        }
        return cls._get_fn_from_table(criterions, name, "criterion") # type: ignore

    @field_validator("optimizer")
    @classmethod
    def get_optimizer(cls, name: str) -> Callable:
        optimizers = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
        }
        return cls._get_fn_from_table(optimizers, name, "optimizer") # type: ignore
    
    @field_validator("metric")
    @classmethod
    def validate_metric(cls, name: str) -> str:
        metrics = cls._get_metrics_dict()
        if name not in metrics:
            raise NotImplementedError(f"metric {name} not yet implemented, must be from {metrics.keys()}")
        return name
    
    @classmethod
    def get_metric(cls, name: str, init_params: dict) -> Callable:
        metrics = cls._get_metrics_dict()
        if name not in metrics:
            raise NotImplementedError(f"metric {name} not yet implemented, must be from {metrics.keys()}")
        return metrics[name](**init_params)

    @staticmethod
    def _get_fn_from_table(fn_table: dict["str", Callable], name: str, errname: str) -> Callable:
        if name not in fn_table:
            raise NotImplementedError(f"{errname} must be one of {fn_table.keys()}, got {name}")
        else:
            return fn_table[name]
    
    @classmethod
    def from_config_file(cls, config_file: str | Path, transforms: Optional[dict[str, Transform | None]] = None):
        config_dict = cls._get_config_dict(config_file)
        transforms_dict = cls._get_transforms_dict(transforms) 
        return cls(**(config_dict | transforms_dict))
    
    @staticmethod
    def _get_metrics_dict() -> dict[str, Callable]:
        return {
            "accuracy": torchmetrics.Accuracy,
            "dice": torchmetrics.F1Score,
            "f1": torchmetrics.F1Score,
            "iou": torchmetrics.JaccardIndex,
            "confusion_matrix": torchmetrics.ConfusionMatrix,
            "cohen_kappa": torchmetrics.CohenKappa,
            "auroc": torchmetrics.AUROC,
        }

    @staticmethod 
    def _get_config_dict(config_file: str | Path) -> dict[str, Any]:
        config_file = get_valid_file_err(config_file, valid_extns=(".yaml",)) 
        with open(config_file, 'r') as config:
            return yaml.safe_load(config)
    
    @staticmethod
    def _get_transforms_dict(transforms: Optional[dict[str, Transform | None]] = None) -> dict[str, Any]: 
        return {"transforms": transforms}

