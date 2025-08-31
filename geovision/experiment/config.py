from typing import Optional
from collections.abc import Callable

import yaml
import torch
import logging
import importlib
import lightning
import torchmetrics
import pandas as pd
from pathlib import Path

from geovision.data import Dataset, DatasetConfig, DataLoaderConfig
from geovision.models.interfaces import ModelConfig
from geovision.io.local import FileSystemIO as fs

logger = logging.getLogger(__name__)

class ExperimentConfig:
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str | int] = None,
        random_seed: Optional[int] = None,
        trainer_task: Optional[str] = None,
        trainer_params: Optional[dict] = None,
        log_params: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        dataset_params: Optional[dict] = None,
        dataloader_params: Optional[dict] = None,
        model_name: Optional[str] = None,
        model_params: Optional[dict] = None,
        metric_name: Optional[str] = None,
        metric_params: Optional[dict] = None,
        criterion_name: Optional[str] = None,
        criterion_params: Optional[dict] = None,
        optimizer_name: Optional[str] = None,
        optimizer_params: Optional[dict] = None,
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        scheduler_config_params: Optional[dict] = None,
        warmup_scheduler_name: Optional[str] = None,
        warmup_scheduler_params: Optional[dict] = None,
        warmup_steps: Optional[int] = None,
        **kwargs,
    ):
        assert isinstance(project_name, str), f"config error (invalid type), expected :project_name to be str, got {type(project_name)}"
        self.project_name = project_name

        if run_name is None or run_name == "":
            from names_generator import generate_name
            self.run_name = generate_name()
        else:
            assert isinstance(run_name, str) or isinstance(run_name, int), \
                f"config error (invalid type), expected :run_name to be str or int, got {type(run_name)}"
            self.run_name = str(run_name)

        if random_seed is not None:
            assert isinstance(random_seed, int), f"config error (invalid type), expected :random_seed to be int, got {type(random_seed)}"
        self.random_seed = random_seed or 42

        if trainer_task is not None:
            assert trainer_task in ("fit", "validate", "test"), \
                f"config error (invalid value), expected :trainer_task to be one of fit, validate, test or predict, got {trainer_task}"
        self.trainer_task = trainer_task

        if trainer_params is not None:
            assert isinstance(trainer_params, dict), f"config error (invalid type), expected :trainer_params to be dict, got {type(trainer_params)}"
        self.trainer_params = trainer_params
        
        if log_params is not None:
            assert isinstance(log_params, dict), f"config error (invalid type), expected :log_params to be dict, got {type(log_params)}"

            log_every_n_steps = log_params.get("log_every_n_steps")
            assert log_every_n_steps is not None, "config error (missing value), expected :log_params to contain log_every_n_steps(int)"
            assert isinstance(log_every_n_steps, int), \
                f"config error (invalid value), expected :log_params[log_every_n_steps] to be an int, got {type(log_every_n_steps)}"
            self.trainer_params["log_every_n_steps"] = log_params["log_every_n_steps"]

            log_every_n_epochs = log_params.get("log_every_n_epochs")
            assert log_every_n_epochs is not None, "config error (missing value), expected :log_params to contain log_every_n_epochs(int)"
            assert isinstance(log_every_n_epochs, int), \
                f"config error (invalid value), expected :log_params[log_every_n_epochs] to be an int, got {type(log_every_n_epochs)}"
            self.trainer_params["check_val_every_n_epoch"] = log_params["log_every_n_epochs"]

            assert log_params.get("log_to_h5") is not None, "config error (missing value), expected :log_params to contain log_to_h5(bool)"
            assert log_params.get("log_to_csv") is not None, "config error (missing value), expected :log_params to contain log_to_csv(bool)"
            assert log_params.get("log_to_wandb") is not None, "config error (missing value), expected :log_params to contain log_to_wandb(bool)"
            assert log_params.get("log_models") is not None, "config error (missing value), expected :log_params to contain log_models(bool)"
            self.trainer_params["enable_checkpointing"] = log_params["log_models"]

            wandb_init_params = log_params.get("wandb_init_params")
            if wandb_init_params is not None:
                assert isinstance(wandb_init_params, dict) is not None, \
                    f"config error (invalid type), expected :wandb_init_params to be dict, got {type(wandb_init_params)}"
        self.log_params = log_params

        if dataset_name is not None:
            assert isinstance(dataset_params, dict), f"config error (invalid type), expected :dataset_params to be dict, got {type(dataset_params)}"
            self.dataset_constructor = self._get_dataset_constructor(dataset_name)
            self.dataset_config = DatasetConfig(random_seed=self.random_seed, **dataset_params)
        else:
            self.dataset_constructor = None
            self.dataset_config = None
        self.dataset_name = dataset_name

        if dataloader_params is not None:
            assert isinstance(dataloader_params, dict), \
                f"config error (invalid type), expected :dataloader_params to be dict, got {type(dataloader_params)}"
            if self.dataset_name is not None: 
                batch_transform_name = dataset_params.get("batch_transform_name")
                if batch_transform_name is not None:
                    dataloader_params["batch_transform_name"] = batch_transform_name 
                    dataloader_params["batch_transform_params"] = dataset_params.get("batch_transform_params", dict())
                    dataloader_params["batch_transform_params"]["num_classes"] = self.dataset_constructor.num_classes
            self.dataloader_config = DataLoaderConfig(**dataloader_params)
            if isinstance(self.trainer_params, dict):
                self.trainer_params["accumulate_grad_batches"] = self.dataloader_config.gradient_accumulation
        else:
            self.dataloader_config = None

        if metric_name is not None:
            self.metric_constructor = self._get_metric_constructor(metric_name)
            self.metric_params = metric_params or dict()
            if self.metric_params.get("task") is None and self.dataset_constructor is not None:
                self.metric_params["task"] = self._get_metric_task(self.dataset_constructor)
            if self.metric_params.get("num_classes") is None and self.dataset_constructor is not None:
                self.metric_params["num_classes"] = self.dataset_constructor.num_classes
        else:
            self.metric_constructor = None
            self.metric_params = None
        self.metric_name = metric_name

        if model_name is not None:
            assert isinstance(model_params, dict), f"config error (invalid type), expected :model_params to be dict, got {type(model_params)}"

            self.model_name = getattr(importlib.import_module("geovision.models.interfaces"), model_name)
            decoder_params = model_params.get("decoder_params") 
            if decoder_params is not None:
                decoder_params["out_ch"] = self.dataset_constructor.num_classes
            self.model_config = ModelConfig(**model_params)

        else:
            self.model_config = None

        if criterion_name is not None:
            self.criterion_constructor = self._get_criterion_constructor(criterion_name)
            self.criterion_params = criterion_params or dict()
            assert isinstance(self.criterion_params, dict), \
                f"config error (invalid type), expected :criterion_params to be dict, got {type(self.criterion_params)}"
        else:
            self.criterion_constructor = None
            self.criterion_params = None
        self.criterion_name = criterion_name

        if optimizer_name is not None:
            self.optimizer_constructor = self._get_optimizer_constructor(optimizer_name)
            self.optimizer_params = optimizer_params or dict()
            assert isinstance(self.optimizer_params, dict), \
                f"config error (invalid type), expected :optimizer_params to be dict, got {type(self.optimizer_params)}"
        else:
            self.optimizer_constructor = None
            self.optimizer_params = None
        self.optimizer_name = optimizer_name

        if scheduler_name is not None:
            self.scheduler_constructor = self._get_scheduler_constructor(scheduler_name)
            self.scheduler_params = scheduler_params or dict()
            assert isinstance(self.scheduler_params, dict), \
                f"config error (invalid type), expected :scheduler_params to be dict, got {type(self.scheduler_params)}"
        else:
            self.scheduler_constructor = None
            self.scheduler_params = None
        self.scheduler_name = scheduler_name

        if warmup_scheduler_name is not None:
            self.warmup_scheduler_constructor = self._get_scheduler_constructor(warmup_scheduler_name)
            self.warmup_scheduler_params = warmup_scheduler_params or dict()
            assert isinstance(self.warmup_scheduler_params, dict), \
                f"config error (invalid type), expected :warmup_scheduler_params to be dict, got {type(self.warmup_scheduler_params)}"
            if scheduler_name is not None:
                assert isinstance(warmup_steps, int), f"config error (invalid type), expected :warmup_steps to be int, got {type(warmup_steps)}"
            self.warmup_steps = warmup_steps 
        else:
            self.warmup_scheduler_constructor = None
            self.warmup_scheduler_params = None
            self.warmup_steps = None
        self.warmup_scheduler_name = warmup_scheduler_name

        self.scheduler_config_params = scheduler_config_params or dict()
        assert isinstance(self.scheduler_config_params, dict), \
            f"config error (invalid type), expected :scheduler_config_params to be dict, got {type(scheduler_config_params)}"

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        namespace = dict()
        exec(config_dict["dataset_params"]["transforms"], namespace)
        config_dict["dataset_params"]["image_pre"] = namespace["image_pre"]  # type: ignore # noqa: F821
        config_dict["dataset_params"]["target_pre"] = namespace["target_pre"]  # type: ignore # noqa: F821
        config_dict["dataset_params"]["train_aug"] = namespace["train_aug"]  # type: ignore # noqa: F821
        config_dict["dataset_params"]["eval_aug"] = namespace["eval_aug"]  # type: ignore # noqa: F821
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        # TODO: print this as a pretty table :: self -> pd.DataFrame -> tabulate -> str
        out = f"==Experiment Config==\nProject Name: {self.project_name}\nRun Name: {self.run_name}\nRandom Seed: {self.random_seed}\n\n"
        out += f"==Logging Config==\n{"\n".join(str(self.log_params).removeprefix("{").removesuffix("}").split(", "))}\n\n"
        out += f"==Dataset Config==\nDataset: {self.dataset_name} [{self.dataset_constructor}]\n{self.dataset_config}Dataloader Params: {self.dataloader_config}\n\n"
        out += f"==Model Config==\nModel Type: {self.model_name}\nEncoder:{self.model_config.encoder_name} {self.model_config.encoder_params}\nDecoder:{self.model_config.decoder_name} {self.model_config.decoder_params}\n\n"
        out += f"==Task Config==\nTrainer Task: {self.trainer_task}\nTrainer Params: {self.trainer_params}\n\n"
        out += f"==Evaluation Config==\nCriterion: {self.criterion_name} [{self.criterion_constructor}]\nCriterion Params: {self.criterion_params}\n"
        out += f"Metric: {self.metric_name} [{self.metric_constructor}]\nMetric Params: {self.metric_params}\n\n"
        out += f"==Training Config==\nOptimizer: {self.optimizer_name} [{self.optimizer_constructor}]\nOptimizer Params: {self.optimizer_params}\n"
        out += f"LR Scheduler: {self.scheduler_name} [{self.scheduler_constructor}]\nLR Scheduler Params: {self.scheduler_params}\nLR Scheduler Config: {self.scheduler_config_params}\n"
        out += f"Warmup LR Scheduler: {self.warmup_scheduler_name} [{self.warmup_scheduler_constructor}]\nWarmup LR Scheduler Params: {self.warmup_scheduler_params}\n\n"
        return out

    @property
    def experiments_dir(self) -> Path:
        """returns path to (created if non-existant) experiments log dir, ~/experiments/{project_name}/{run_name}"""
        return fs.get_new_dir(Path.home(), "experiments", self.project_name, self.run_name)

    @property
    def ckpt_path(self) -> Optional[Path]:
        if self.model_config.ckpt_path is not None:
            return self.model_config.ckpt_path

        def _version(path: Path) -> int:
            parts = path.stem.split('v')
            if len(parts) > 1:
                return int(parts[-1])
            return 1
        
        ckpts_df = pd.DataFrame({"ckpt_path": self.experiments_dir.rglob("*.ckpt")})
        ckpts_df["epoch"] = ckpts_df["ckpt_path"].apply(lambda x: int(x.stem.split('_')[0].removeprefix("epoch=")))
        ckpts_df["step"] = ckpts_df["ckpt_path"].apply(lambda x: int(x.stem.split('_')[1].removeprefix("step=").split('v')[0]))
        ckpts_df["version"] = ckpts_df["ckpt_path"].apply(_version)
        ckpts_df = ckpts_df.sort_values(["epoch", "step", "version"]).reset_index(drop = True)
        print(ckpts_df)

        if len(ckpts_df) == 0:
            return None
        return ckpts_df.iloc[-1]["ckpt_path"]

    @property
    def wandb_init_params(self) -> dict:
        params = {"project": self.project_name, "name": self.run_name, "dir": self.experiments_dir, "resume": "never"} 
        #params.update(self.log_params.get("wandb_params", dict()))
        return params | self.log_params.get("wandb_init_params", dict())

    @property
    def grad_accum(self) -> int:
        return self.dataloader_config.gradient_accumulation

    def get_metric(self, metric_name: Optional[str] = None, addl_metric_params: Optional[dict] = None) -> torchmetrics.Metric:
        metric_name: str = metric_name or self.metric_name
        metric_params: dict = self.metric_params or dict()
        return self._get_metric_constructor(metric_name)(**(metric_params | (addl_metric_params or dict())))

    def _get_metric_task(self, dataset: Callable) -> str:
        if dataset.task == "classification":
            if dataset.subtask == "multiclass":
                return "multiclass" if dataset.num_classes > 2 else "binary"
            elif dataset.subtask == "multilabel":
                return "multilabel"
        elif dataset.task == "segmentation":
            #return "multiclass" if dataset.num_classes > 2 else "binary"
            return "multiclass"
        else:
            raise AssertionError(f"config error (invalid value), {dataset.task} is invalid")

    def _get_model_constructor(self, model_name) -> Callable[..., torch.nn.Module]:
        module, class_name = model_name.split('.')
        module = '.'.join(["geovision", "data", module])
        return getattr(importlib.import_module(module), class_name)
        
    def _get_metric_constructor(self, metric_name: str) -> Callable[..., torchmetrics.Metric]:
        return getattr(importlib.import_module("torchmetrics"), metric_name)
        
    def _get_dataset_constructor(self, dataset_name: str) -> Callable[..., Dataset]:
        module, class_name = dataset_name.split('.')
        module = '.'.join(["geovision", "data", module])
        return getattr(importlib.import_module(module), class_name)

    def _get_criterion_constructor(self, criterion_name: str) -> Callable[..., torch.nn.Module]:
        return getattr(importlib.import_module("torch.nn"), criterion_name)
        
    def _get_optimizer_constructor(self, optimizer_name: str) -> Callable[..., torch.optim.Optimizer]:
        return getattr(importlib.import_module("torch.optim"), optimizer_name)
        
    def _get_scheduler_constructor(self, scheduler_name: str) -> Callable[..., torch.optim.lr_scheduler.LRScheduler]:
        return getattr(importlib.import_module("torch.optim.lr_scheduler"), scheduler_name)
        
if __name__ == "__main__":
    config = ExperimentConfig.from_yaml(Path.home() / "dev" / "geovision" / "geovision" / "scripts" / "config.yaml")
    #print(config.ckpt_path)
    print(config)
