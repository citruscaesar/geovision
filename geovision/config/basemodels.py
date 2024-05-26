from typing import Any, Callable, Optional

import yaml # type: ignore
import torch
import torchmetrics
import names_generator
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator 
from torchvision.transforms.v2 import Transform # type: ignore
from geovision.io.local import get_valid_file_err
from geovision.data.dataset import Dataset, DatasetConfig, TransformsConfig

from .parsers import (
    get_dataset,
    get_model,
    get_criterion,
    get_optimizer,
    get_metric
) 

class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    gradient_accumulation: int
    persistent_workers: bool
    pin_memory: bool

class NNConfig(BaseModel):
    weights: str
    dropout: float | None = None
    num_layers: int | None = None

class CriterionParams(BaseModel):
    reduction: str 
    weight: tuple | None = None
    label_smoothing: float | None = None
    ignore_index: int | None = None

    #CTCLoss
    blank: int | None = None
    zero_infinity: bool | None = None
    #NLLLoss (Poisson, Gaussian)
    log_input: bool | None = None
    eps: bool | None = None
    full: bool | None = None
    #KLDiv
    log_target: bool | None = None
    #BCEWithLogitsLoss
    pos_weight: tuple | None = None
    #MarginLoss (Multi, Triplet)
    p: int | None = None
    margin: float | None = None
    swap: bool | None = None
    #HuberLoss
    delta: float | None = None
    #SmoothL1Loss
    beta: float | None = None
    distance_function: str | None = None

class OptimizerParams(BaseModel):
    lr: float
    foreach: bool | None = None
    maximize: bool | None = None
    capturable: bool | None = None
    differentiable: bool | None = None
    fused: bool | None = None

    #SGD
    momentum: float | None = None
    weight_decay: float | None = None
    dampening: float | None = None
    nesterov: bool | None = None

    #Adam
    betas: tuple | None = None
    eps: float | None = None
    amsgrad: bool | None = None

    rho: float | None = None
    lr_decay: float | None = None

    #ASGD
    lambd: float | None = None
    alpha: float | None = None
    t0: float | None = None

    #LBFGS
    max_iter: int | None = None
    max_eval: int| None = None
    tolerance_grad: float | None = None
    tolerance_change: float | None = None
    history_search: int | None = None
    line_search_fn: str | None = None

    #NAdam
    momentum_decay: float | None = None
    decoupled_weight_decay: bool | None = None

    #RMSProp
    centered: bool | None = None

    #RProp
    etas: tuple | None = None
    step_sizes: tuple | None = None

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True)
    name: str | None = None
    dataset: Dataset | str
    dataset_root: Path | str
    dataset_df: Path | str | None = None
    dataset_params: DatasetConfig | None = None
    transforms: TransformsConfig | None = None
    dataloader_params: DataLoaderConfig
    nn: torch.nn.Module | str
    nn_params: NNConfig 
    criterion: torch.nn.Module | str
    criterion_params: CriterionParams 
    optimizer: torch.optim.Optimizer | str
    optimizer_params: OptimizerParams 
    metric: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str | None = None) -> str:
        if name is None or name == "":
            return names_generator.generate_name() 
        return name

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, name: str) -> Dataset:
        return get_dataset(name)
    
    @field_validator("dataset_root")
    @classmethod
    def validate_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"dataset root not found on local fs, got {root}")
        return root

    @field_validator("dataset_df")
    @classmethod
    def validate_dataset_df(cls, df: str | Path | None = None) -> Path | None:
        return get_valid_file_err(df, valid_extns=(".csv",)) if df is not None else None

    @field_validator("nn")
    @classmethod
    def validate_nn(cls, name: str) -> Callable:
        return get_model(name)

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, name: str) -> Callable:
        return get_criterion(name)
        
    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, name: str) -> Callable:
        return get_optimizer(name)
    
    @field_validator("metric")
    @classmethod
    def validate_metric(cls, name: str) -> str:
        get_metric(name)
        return name
    
    @classmethod
    def get_metric(cls, name: str, metric_params: dict[str, str | int]) -> torchmetrics.Metric:
        return get_metric(name)(**metric_params) 

    @classmethod
    def from_yaml(cls, config_file: str | Path, transforms: Optional[dict[str, Transform | None]] = None):

        def _get_config_dict(config_file: str | Path) -> dict[str, Any]:
            config_file = get_valid_file_err(config_file, valid_extns=(".yaml",)) 
            with open(config_file, 'r') as config:
                return yaml.safe_load(config)

        def _get_transforms_dict(transforms: Optional[dict[str, Transform | None]] = None) -> dict[str, Any]: 
            return {"transforms": transforms}

        config_dict = _get_config_dict(config_file)
        transforms_dict = _get_transforms_dict(transforms) 
        return cls(**(config_dict | transforms_dict))