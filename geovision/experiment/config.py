from types import ModuleType
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING


import yaml
import copy
import torch
import logging
import importlib
import lightning
import torchmetrics
import pandas as pd
from pathlib import Path
from names_generator import generate_name

from dataclasses import dataclass
from geovision.io.local import FileSystemIO as fs
from torch.utils.data import default_collate
from torchvision.transforms.v2 import Transform

logger = logging.getLogger(__name__)

class Builder:
    name: str
    src: str
    kwargs: dict[str, Any]
    module: ModuleType
    constructor: Callable[..., Any]

    def __init__(self, name: str, src: str, kwargs: dict[str, Any]):

        assert isinstance(name, str), f"config error, expected :name={name} to be of type str, got type={type(name)}"
        self.name = name

        assert isinstance(src, str), f"config error, expected :src={src} to be of type str, got type={type(src)}"
        self.src = src

        assert isinstance(kwargs, dict), f"config error, expected :kwargs={kwargs} to be of type dict, got type={type(kwargs)}"
        self.kwargs = kwargs

        self.module = importlib.import_module(src)
        self.constructor: Callable[..., Any] = getattr(self.module, name)
        assert callable(self.constructor)

    def build(self):
        return self.constructor(**self.kwargs)

@dataclass(eq = False, kw_only=True)
class DatasetConfigParams:
    random_seed: int
    df: Optional[pd.DataFrame]
    tabular_sampler: Optional[Builder]
    spatial_sampler: Optional[Builder]
    spectral_sampler: Optional[Builder]
    temporal_sampler: Optional[Builder]
    image_pre: Transform
    target_pre: Transform
    train_aug: Transform
    eval_aug: Transform

class DatasetBuilder(Builder):
    # constructor: Callable[..., Dataset]
    # config: DatasetConfigParams 

    def __init__(self, name: str, src: str, kwargs: dict[str, Any], random_seed: int):
        super().__init__(name, src, kwargs)

        assert isinstance(self.kwargs, dict), f"error, DatasetConfigParams.kwargs is not dict, its {type(self.kwargs)}"
        self.kwargs["random_seed"] = random_seed
        
        if df := self.kwargs.get("df") is not None:
            assert isinstance(df, str), f"config error, expected dataset.kwargs.df to be of type str (local path) or None, got {type(df)}"

            df: Path = fs.get_valid_file_err(df)
            if df.suffix == ".csv":
                try:
                    df = pd.read_csv(df)
                except Exception as e:
                    logger.error(f"got error {e} when trying to read dataframe from csv: {df}")

            elif df.suffix == ".h5":
                try:
                    df = pd.read_hdf(df, mode = "r", key = "index")
                except Exception as e:
                    logger.error(f"got error {e} when trying to read dataframe from hdf5: {df}::index")

            elif df.suffix == ".parquet":
                try:
                    df = pd.read_parquet(df)
                except Exception as e:
                    logger.error(f"got error {e} when trying to read dataframe from parquet: {df}")
            else:
                raise AssertionError(f"expected dataset.kwargs.df as one of .csv, .h5 or .parquet, got {df.suffix}")

            assert isinstance(df, pd.DataFrame)
            self.kwargs["df"] = df
        
        for sampler_name in ("tabular", "spatial", "spectral", "temporal"):
            sampler = self.kwargs.get(f"{sampler_name}_sampler")
            if sampler is not None:  # sampler should be of Builder layout (i.e. dict with a name: str, src: str and kwargs: dict[str, Any])
                assert isinstance(sampler, dict), f"config error, expected dataset.kwargs.tabular_sampler to be of type dict or None, got {type(sampler)}"
                sampler["kwargs"] = (sampler["kwargs"] or dict()) | {"random_seed": self.kwargs["random_seed"]} # Error if sampler["kwargs"] is not dict nor none.
                self.kwargs[f"{sampler_name}_sampler"] = Builder(**sampler)
       
        transforms = self.kwargs.get("transforms")
        if transforms is not None:
            assert isinstance(transforms, str), f"config error, expected dataset.kwargs.transforms to be of type str or None, got {type(transforms)}"

            namespace = dict()
            exec(transforms, namespace)
            self.kwargs.update({k:v for k,v in namespace.items() if isinstance(v, Transform)})
            self.kwargs.pop("transforms", None)

        self.config = DatasetConfigParams(**self.kwargs)
       
    def build(self, split: Optional[str] = None): # -> Dataset
        return self.constructor(split = split, config = self.config)

    @property
    def torchmetrics_task(self) -> str:
        if self.constructor.task == "classification":
            if self.constructor.subtask == "multiclass":
                return "multiclass" if self.constructor.num_classes > 2 else "binary"
            elif self.constructor.subtask == "multilabel":
                return "multilabel"
        elif self.constructor.task == "segmentation":
            if self.constructor.subtask in ("semantic", "instance", "panoptic"):
                return "multiclass"
            #return "multiclass" if self.constructor.num_classes > 2 else "binary"
        else:
            raise AssertionError(f"config error (invalid value), {self.constructor.task} is invalid")

class DataLoaderConfigParams:
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            persistent_workers: Optional[bool] = None,
            pin_memory: Optional[bool] = None,
            prefetch_factor: Optional[int] = None,
            gradient_accumulation: int = 1,
            batch_transform: Optional[Builder] = None,
    ):
        # Validate batch_size
        assert isinstance(batch_size, int), \
            f"config error (invalid type), expected :batch_size to be int, got {type(batch_size)}"
        assert batch_size > 0, \
            f"config error (invalid value), expected :batch_size to be positive, got {batch_size}"

        # Validate gradient_accumulation
        assert isinstance(gradient_accumulation, int), \
            f"config error (invalid type), expected :gradient_accumulation to be int, got {type(gradient_accumulation)}"
        assert gradient_accumulation > 0, \
            f"config error (invalid value), expected :gradient_accumulation to be positive, got {gradient_accumulation}"

        self.batch_size = batch_size // gradient_accumulation
        self.gradient_accumulation = gradient_accumulation

        # Validate num_workers
        assert isinstance(num_workers, int), \
            f"config error (invalid type), expected :num_workers to be int, got {type(num_workers)}"
        assert num_workers >= 0, \
            f"config error (invalid value), expected :num_workers to be non-negative, got {num_workers}"
        self.num_workers = num_workers

        # Validate persistent_workers (requires num_workers > 0)
        if persistent_workers is not None:
            assert isinstance(persistent_workers, bool), \
                f"config error (invalid type), expected :persistent_workers to be bool, got {type(persistent_workers)}"
            if persistent_workers:
                assert num_workers > 0, \
                    f"config error (invalid value), :persistent_workers=True requires num_workers > 0, got num_workers={num_workers}"
        self.persistent_workers = persistent_workers

        # Validate pin_memory
        if pin_memory is not None:
            assert isinstance(pin_memory, bool), \
                f"config error (invalid type), expected :pin_memory to be bool, got {type(pin_memory)}"
        self.pin_memory = pin_memory

        # Validate prefetch_factor (requires num_workers > 0)
        if prefetch_factor is not None:
            assert isinstance(prefetch_factor, int), \
                f"config error (invalid type), expected :prefetch_factor to be int, got {type(prefetch_factor)}"
            assert prefetch_factor > 0, \
                f"config error (invalid value), expected :prefetch_factor to be positive, got {prefetch_factor}"
            assert num_workers > 0, \
                f"config error (invalid value), :prefetch_factor requires num_workers > 0, got num_workers={num_workers}"
        self.prefetch_factor = prefetch_factor

        # Validate and configure batch_transform
        self.collate_fn: Optional[Callable] = None 
        if batch_transform is not None:
            self.collate_fn = lambda batch: batch_transform.build()(*default_collate(batch))

    @property
    def kwargs(self) -> dict[str, int | Callable]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "collate_fn": self.collate_fn
        }

class MetricBuilder(Builder):
    key:str
    def __init__(self, name: str, src: str = "torchmetrics", kwargs: dict[str, Any] = dict(), key: Optional[str] = None):
        super().__init__(name, src, kwargs)
        if key is not None:
            assert isinstance(key, str), f"config error, expected :key={key} to be of type str, got type={type(key)}"
            self.key = key
        else:
            self.key = name
    
class ExperimentConfig:
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str | int] = None,
        random_seed: Optional[int] = None,
        trainer_kwargs: Optional[dict] = None,
        logger_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,

        dataset: Optional[Builder] = None,
        module: Optional[Builder] = None,
        metrics: Optional[list[Builder]] = None,
        criterion: Optional[Builder] = None,
        optimizer: Optional[Builder] = None,
        schedulers: Optional[list[Builder]] = None,
        scheduler_intervals: Optional[list[int]] = None,
        scheduler_kwargs: Optional[dict[str, Any]] = None,
    ):

        assert isinstance(project_name, str), f"config error (invalid type), expected :project_name to be str, got {type(project_name)}"
        self.project_name = project_name

        if run_name is None or run_name == "":
            self.run_name = generate_name()
        else:
            assert isinstance(run_name, str) or isinstance(run_name, int), \
                f"config error (invalid type), expected :run_name to be str or int, got {type(run_name)}"
            self.run_name = str(run_name)

        self.random_seed = 42
        if random_seed is not None:
            assert isinstance(random_seed, int), f"config error (invalid type), expected :random_seed to be int, got {type(random_seed)}"
            self.random_seed = random_seed

        self.trainer_kwargs = None
        if trainer_kwargs is not None:
            assert isinstance(trainer_kwargs, dict), f"config error (invalid type), expected :trainer_kwargs to be dict, got {type(trainer_kwargs)}"
            self.trainer_kwargs = trainer_kwargs 

        if logger_kwargs is not None:
            assert isinstance(logger_kwargs, dict), f"config error (invalid type), expected :logger_kwargs to be dict, got {type(logger_kwargs)}"

            log_every_n_steps = logger_kwargs.get("log_every_n_steps")
            assert log_every_n_steps is not None, "config error (missing value), expected :logger_kwargs to contain log_every_n_steps(int)"
            assert isinstance(log_every_n_steps, int), \
                f"config error (invalid value), expected :logger_kwargs[log_every_n_steps] to be an int, got {type(log_every_n_steps)}"
            self.trainer_kwargs["log_every_n_steps"] = logger_kwargs["log_every_n_steps"]

            log_every_n_epochs = logger_kwargs.get("log_every_n_epochs")
            assert log_every_n_epochs is not None, "config error (missing value), expected :logger_kwargs to contain log_every_n_epochs(int)"
            assert isinstance(log_every_n_epochs, int), \
                f"config error (invalid value), expected :logger_kwargs[log_every_n_epochs] to be an int, got {type(log_every_n_epochs)}"
            self.trainer_kwargs["check_val_every_n_epoch"] = logger_kwargs["log_every_n_epochs"]

            for log_tgt in ("h5", "csv", "wandb", "tensorboard"):
                flag = logger_kwargs.get(f"log_to_{log_tgt}")
                assert flag is not None, f"config error (missing value), expected logger_kwargs.log_to_{log_tgt} to be defined"
                assert isinstance(flag, bool), f"config error (missing value), expected logger_kwargs.log_to_{log_tgt} to be of type bool, got {type(flag)}"

            enable_ckpt_flag = logger_kwargs.get("log_models")
            assert enable_ckpt_flag is not None, "config error (missing value), expected :logger_kwargs.log_models to be defined"
            assert isinstance(enable_ckpt_flag, bool), f"config error (invalid type), expected :logger_kwargs.log_models to be of type bool, got {type(enable_ckpt_flag)}"
            self.trainer_kwargs["enable_checkpointing"] = enable_ckpt_flag 

            wandb_init_kwargs = logger_kwargs.get("wandb_init_kwargs")
            if wandb_init_kwargs is not None:
                assert isinstance(wandb_init_kwargs, dict), f"config error (invalid type), expected :wandb_init_kwargs to be dict, got {type(wandb_init_kwargs)}"
        self.logger_kwargs = logger_kwargs

        self.dataset = None
        if dataset is not None:
            assert isinstance(dataset, dict), f"config error (invalid type), expected :dataset to be dict, got {type(dataset)}"

            kwargs = dataset.get("kwargs")
            assert isinstance(kwargs, dict)

            batch_transform = None
            if "batch_transform" in kwargs:
                batch_transform = kwargs["batch_transform"]
                del kwargs["batch_transform"]

            self.dataset = DatasetBuilder(random_seed = self.random_seed, **dataset)

            if batch_transform is not None:
                assert batch_transform.get("name") in ("CutMix", "MixUp"), \
                    f"config error (not implemented), expected :batch_transform.name to be one of CutMix, MixUp, got {batch_transform.name}"
                batch_transform["kwargs"] = batch_transform.get("kwargs") or dict()
                batch_transform["kwargs"]["num_classes"] = self.dataset.constructor.num_classes
                dataloader_kwargs["batch_transform"] = Builder(**batch_transform)
            else:
                dataloader_kwargs["batch_transform"] = None 
                        
        self.dataloader_config = None
        if dataloader_kwargs is not None:
            assert isinstance(dataloader_kwargs, dict), \
                f"config error (invalid type), expected :dataloader_kwargs to be dict, got {type(dataloader_kwargs)}"
            self.dataloader_config = DataLoaderConfigParams(**dataloader_kwargs)

            if self.trainer_kwargs is not None:
                self.trainer_kwargs["accumulate_grad_batches"] = self.dataloader_config.gradient_accumulation
        
        self.metrics: list[Builder] = list()
        if metrics is not None:
            assert isinstance(metrics, list), "config error, expected :metrics to be a list" 

            for metric in metrics:
                assert isinstance(metric, dict)
                metric = MetricBuilder(**metric)
                if metric.src == "torchmetrics":
                    metric.kwargs = {"task": self.dataset.torchmetrics_task, "num_classes": self.dataset.constructor.num_classes} | metric.kwargs
                elif metric.src == "cocoeval":
                    ...
                self.metrics.append(metric)

        # to instantitate the LightningModule, we don't create a builder class. 
        # instead, the responsibility for validating the kwargs, importing the modules, and creating the models is left to the LightningModule itself. 
        # this is done because the layout of the configuration is heavily dependent on the type of module, so dealing with them abstractly seems like a bad idea
        # thus a copy of the entire ExperimentConfig is passed to the LightningModule, instead

        self.module_constructor = None
        self.module_kwargs = None

        if module is not None:
            assert isinstance(module, dict), "config error, expected :module to be of type dict " 
            
            module = Builder(**module)
            self.module_constructor = module.constructor

            ckpt_path = module.kwargs.get("ckpt_path")
            if ckpt_path is not None:
                ckpt_path = fs.get_valid_file_err(ckpt_path)
                if self.trainer_kwargs is not None: # if there is a trainer, pass ckpt_path to it
                    assert self.trainer_kwargs.get("ckpt_path") is None, "config error, ckpt_path provided in both trainer_kwargs and module.kwargs"
                    self.trainer_kwargs["ckpt_path"] = ckpt_path
                else: # if there is no trainer, idk what to do 
                    self.module_constructor = self.module_constructor.load_from_checkpoint
                    module.kwargs["ckpt_path"] = ckpt_path

            submodule_keys = (k for k in module.kwargs.keys() if k.startswith("model_"))
            for key in submodule_keys:
                submodule = module.kwargs.get(key)
                assert isinstance(submodule, dict), f"config error, expected :module.kwargs.{key} to be of type dict, got {type(submodule)}" 
                module.kwargs[key] = Builder(**submodule)
            
            self.module_kwargs = copy.deepcopy(module.kwargs)

        # # NOTE: DO NOT REMOVE
        # # if model_name is not None:
            # # assert isinstance(model_kwargs, dict), f"config error (invalid type), expected :model_kwargs to be dict, got {type(model_kwargs)}"
            # # self.model_name = getattr(importlib.import_module("geovision.models.interfaces"), model_name)
            # # decoder_kwargs = model_kwargs.get("decoder_kwargs") 
            # # if decoder_kwargs is not None:
                # # decoder_kwargs["out_ch"] = self.dataset_constructor.num_classes
            # # self.model_config = ModelConfig(**model_kwargs)
        # # else:
            # # self.model_config = None

        self.criterion = None
        if criterion is not None:
            assert isinstance(criterion, dict)
            if criterion.get("kwargs") is None:
                criterion["kwargs"] = dict() 
            self.criterion = Builder(**criterion)
        
        self.optimizer = None
        if optimizer is not None:
            assert isinstance(optimizer, dict)
            self.optimizer = Builder(**optimizer)
        
        # # NOTE: read from lightning module configure schedulers documentation

        self.schedulers: Optional[list[Builder]] = None 
        if schedulers is not None:
            assert isinstance(schedulers, list)

            for scheduler in schedulers:
                assert isinstance(scheduler, dict)
                scheduler = Builder(**scheduler)
            
            self.schedulers = schedulers
        
        self.scheduler_intervals: Optional[list[int]] = None
        if scheduler_intervals is not None:
            assert isinstance(scheduler_intervals, list)
            assert len(scheduler_intervals) == len(self.schedulers) - 1
            for interval in scheduler_intervals:
                assert isinstance(interval, int)
            self.scheduler_intervals = scheduler_intervals
        
        self.scheduler_kwargs = None
        if scheduler_kwargs is not None:
            assert isinstance(scheduler_kwargs, dict)
            self.scheduler_kwargs = scheduler_kwargs

        # if scheduler_name is not None:
            # self.scheduler_constructor = self._get_scheduler_constructor(scheduler_name)
            # self.scheduler_kwargs = scheduler_kwargs or dict()
            # assert isinstance(self.scheduler_kwargs, dict), \
                # f"config error (invalid type), expected :scheduler_kwargs to be dict, got {type(self.scheduler_kwargs)}"
        # else:
            # self.scheduler_constructor = None
            # self.scheduler_kwargs = None
        # self.scheduler_name = scheduler_name

        # if warmup_scheduler_name is not None:
            # self.warmup_scheduler_constructor = self._get_scheduler_constructor(warmup_scheduler_name)
            # self.warmup_scheduler_kwargs = warmup_scheduler_kwargs or dict()
            # assert isinstance(self.warmup_scheduler_kwargs, dict), \
                # f"config error (invalid type), expected :warmup_scheduler_kwargs to be dict, got {type(self.warmup_scheduler_kwargs)}"
            # if scheduler_name is not None:
                # assert isinstance(warmup_steps, int), f"config error (invalid type), expected :warmup_steps to be int, got {type(warmup_steps)}"
            # self.warmup_steps = warmup_steps 
        # else:
            # self.warmup_scheduler_constructor = None
            # self.warmup_scheduler_kwargs = None
            # self.warmup_steps = None
        # self.warmup_scheduler_name = warmup_scheduler_name

        # self.scheduler_config_kwargs = scheduler_config_kwargs or dict()
        # assert isinstance(self.scheduler_config_kwargs, dict), \
            # f"config error (invalid type), expected :scheduler_config_kwargs to be dict, got {type(scheduler_config_kwargs)}"

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        return cls(**config_dict)
    
    # def __repr__(self) -> str:
        # # TODO: print this as a pretty table :: self -> pd.DataFrame -> tabulate -> str
        # out = f"==Experiment Config==\nProject Name: {self.project_name}\nRun Name: {self.run_name}\nRandom Seed: {self.random_seed}\n\n"
        # out += f"==Logging Config==\n{"\n".join(str(self.log_kwargs).removeprefix("{").removesuffix("}").split(", "))}\n\n"
        # out += f"==Dataset Config==\nDataset: {self.dataset_name} [{self.dataset_constructor}]\n{self.dataset_config}Dataloader Params: {self.dataloader_config}\n\n"
        # out += f"==Model Config==\nModel Type: {self.model_name}\nEncoder:{self.model_config.encoder_constructor.__qualname__} {self.model_config.encoder_kwargs}\nDecoder:{self.model_config.decoder_constructor.__qualname__} {self.model_config.decoder_kwargs}\n\n"
        # out += f"==Task Config==\nTrainer Task: {self.trainer_task}\nTrainer Params: {self.trainer_kwargs}\n\n"
        # out += f"==Evaluation Config==\nCriterion: {self.criterion_name} [{self.criterion_constructor}]\nCriterion Params: {self.criterion_kwargs}\n"
        # out += f"Metric: {self.metric_name} [{self.metric_constructor}]\nMetric Params: {self.metric_kwargs}\n\n"
        # out += f"==Training Config==\nOptimizer: {self.optimizer_name} [{self.optimizer_constructor}]\nOptimizer Params: {self.optimizer_kwargs}\n"
        # out += f"LR Scheduler: {self.scheduler_name} [{self.scheduler_constructor}]\nLR Scheduler Params: {self.scheduler_kwargs}\nLR Scheduler Config: {self.scheduler_config_kwargs}\n"
        # out += f"Warmup LR Scheduler: {self.warmup_scheduler_name} [{self.warmup_scheduler_constructor}]\nWarmup LR Scheduler Params: {self.warmup_scheduler_kwargs}\n\n"
        # return out

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

        if len(ckpts_df) > 0:
            print(ckpts_df)
        else:
            print("no ckpts found in experiment logs dir")

        if len(ckpts_df) == 0:
            return None
        return ckpts_df.iloc[-1]["ckpt_path"]

    @property
    def wandb_init_kwargs(self) -> dict:
        kwargs = {"project": self.project_name, "name": self.run_name, "dir": self.experiments_dir, "resume": "never"} 
        #kwargs.update(self.log_kwargs.get("wandb_kwargs", dict()))
        return kwargs | self.log_kwargs.get("wandb_init_kwargs", dict())

    @property
    def grad_accum(self) -> int:
        return self.dataloader_config.gradient_accumulation
    
    @staticmethod
    def _is_builder(block: dict[str, ...]) -> bool:
        if isinstance(block, dict):
            reqd_keys = set(["name", "src", "kwargs"])
            if set(block.keys()).intersection(reqd_keys) == reqd_keys: # all required keys are in the dict 
                return True
        return False

if __name__ == "__main__":
    config = ExperimentConfig.from_yaml(Path.home() / "dev" / "geovision" / "geovision" / "scripts" / "config.yaml")
    #print(config.ckpt_path)
    print(config)
