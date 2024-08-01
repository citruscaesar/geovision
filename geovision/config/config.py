from typing import Callable, Optional, Literal

import yaml # type: ignore
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from pathlib import Path
from pydantic import BaseModel, field_validator 
from geovision.io.local import get_new_dir 
from geovision.data.dataset import Dataset
from geovision.data.config import DatasetConfig, DataLoaderConfig

from .parsers import (
    validate_dataset_err, validate_criterion_err, validate_scheduler_err, validate_optimizer_err, validate_model_err, validate_metric_err,
    parse_dataset, parse_criterion, parse_scheduler, parse_optimizer, parse_model, parse_metric
)

class ExperimentConfig(BaseModel):
    name: str 
    random_seed: int
    metric: str
    dataset: str
    nn: str
    criterion: str
    optimizer: str
    scheduler: str | None = None
    trainer_task: Literal["fit", "validate", "test"]

    dataset_params: DatasetConfig
    dataloader_params: DataLoaderConfig
    nn_params: dict 
    criterion_params: dict
    optimizer_params: dict
    scheduler_params: dict | None = None
    metric_params: dict | None = None
    log_params: dict 
    trainer_params: dict

    @property
    def dataset_(self) -> Callable:
        return parse_dataset(self.dataset)

    @property
    def nn_(self) -> Callable:
        return parse_model(self.nn)

    @property
    def criterion_(self) -> Callable:
        return parse_criterion(self.criterion)

    @property
    def optimizer_(self) -> Callable:
        return parse_optimizer(self.optimizer)

    @property
    def scheduler_(self) -> Callable | None:
        if self.scheduler is not None:
            return parse_scheduler(self.scheduler)
        return None
    
    @property
    def metric_(self) -> Callable:
        return parse_metric(self.metric)

    @property 
    def experiments_dir(self) -> Path:
        """generates and returns path to experiments log dir, ~/experiments/{ds_name}_{ds_task}/{config.name}"""
        ds_name, _, ds_task = self.dataset_.name.split('_')
        expr_name = self.name.replace(' ', '_')
        return get_new_dir(Path.home(), "experiments", '_'.join([ds_name, ds_task]), expr_name)
    
    def get_dataset(self, split: str) -> Dataset:
        return self.dataset_(split = split, config = self.dataset_params)
    
    def get_nn(self) -> Module:
        return self.nn_(**({"num_classes": self.dataset_.num_classes} | self.nn_params))
    
    def get_criterion(self) -> Module:
        return self.criterion_(**self.criterion_params)
    
    def get_optimizer(self, model_params) -> Optimizer:
        return self.optimizer_(model_params, **self.optimizer_params)
    
    def get_scheduler(self, optimizer) -> LRScheduler | None:
        if self.scheduler_ is None:
            return None
        return self.scheduler_(optimizer, **self.scheduler_params)
    
    def get_dataloader_params(self) -> DataLoader:
        return {
            "batch_size": self.dataloader_params.batch_size // self.dataloader_params.gradient_accumulation,
            "num_workers": self.dataloader_params.num_workers,
            "persistent_workers": self.dataloader_params.persistent_workers,
            "pin_memory": self.dataloader_params.pin_memory
        }
         
    def get_metric_params(self) -> dict:
        """get torchmetrics based on config.dataset and config.metric_params"""
        match (t := self.dataset_.name.split('_')[-1]):
            case c if c in ("classification", "segmentation"):
                task = "multiclass" if self.dataset_.num_classes > 2 else "binary"
            case "multilabelclassification":
                task = "multilabel"
            case _:
                raise ValueError(f"invalid dataset task, got {t}")

        metric_params = {"task": task, "num_classes": self.dataset_.num_classes}
        if self.metric_params is not None:
            metric_params.update(self.metric_params)
        return metric_params
    
    def get_metric(self, metric_name: Optional[str] = None, addl_params: Optional[dict] = None) -> Metric:
        """get a torchmetric instance using config.dataset and config.metric_params, :metric_name defaults to config.metric, raises NotImplementedError if :metric_name is not found by parser"""
        metric_params = self.get_metric_params() 
        if addl_params is not None:
            metric_params.update(addl_params)
        if metric_name is not None:
            validate_metric_err(metric_name)
            metric = parse_metric(metric_name)
        else:
            metric = self.metric_
        return metric(**metric_params)

    def get_wandb_init_params(self) -> dict:
        ds_name, _, ds_task = self.dataset.split('_')
        params = {
            "project": f"{ds_name}_{ds_task}", 
            "name": self.name, "resume": "never", 
            "config": self.model_dump(),
            "dir": self.experiments_dir,
        }
        params.update(self.log_params.get("wandb_params", dict()))
        return params
    
    @classmethod
    def from_config(cls, config_file: Path):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config["dataset_params"]["random_seed"] = config["random_seed"]
        return cls.model_validate(config) 

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        if name.replace(' ', '') == '':
            from names_generator import generate_name
            return generate_name() 
        return name

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, name: str) -> str:
        validate_dataset_err(name)
        return name
    
    @field_validator("nn")
    @classmethod
    def _validate_nn(cls, name: str) -> str:
        validate_model_err(name)
        return name

    @field_validator("criterion")
    @classmethod
    def _validate_criterion(cls, name: str) -> str:
        validate_criterion_err(name)
        return name 

    @field_validator("scheduler")
    @classmethod
    def _validate_scheduler(cls, name: str | None) -> str | None:
        if name == '' or name is None:
            return None
        else:
            validate_scheduler_err(name)
            return name
        
    @field_validator("optimizer")
    @classmethod
    def _validate_optimizer(cls, name: str) -> str:
        validate_optimizer_err(name)
        return name
    
    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, name: str) -> str:
        validate_metric_err(name)
        return name