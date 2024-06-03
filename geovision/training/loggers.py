from typing import Any, Mapping, Literal, Optional
from numpy.typing import NDArray

import h5py
import torch
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import count
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from geovision.config.basemodels import ExperimentConfig
from geovision.data.dataset import Dataset
from geovision.logging import get_logger
from geovision.io.local import get_new_dir, get_experiments_dir
from geovision.analysis.viz import plot_confusion_matrix, plot_metrics_table
from .metrics import get_classification_metrics_df, get_classification_metrics_dict

class ExperimentWriter:
    # experiment.h5/
    #   run={run_idx: uint32}/
    #       epoch_begin: uint32 
    #       step_begin: uint32 
    #       log_every_n_steps: uint32 
    #       "{split}_{metric}_step": array[float32, (num_steps,)] 
    #       "{split}_{metric}_epoch": array[float32, (num_epochs,)] 
    #       val_confusion_matrix_epoch: array[uint32, (num_epochs, num_classes, num_classes)]
    #       val_logits_epoch: array[float32, (num_epochs, num_samples_per_epoch[split], num_classes)] 


    def __init__(self, root: Path):
        self._root = root / "experiment.h5"
        self._logs = h5py.File(self._root, mode = 'a')
        with self._logs as logfile:
            self._run_idxs = sorted(int(key.removeprefix("run=")) for key in logfile.keys())
            if len(self._run_idxs) == 0:
                self._run = -1
            self._run = self._run_idxs[-1]
    
    @property
    def run_idxs(self) -> list[int]:
        return self._run_idxs
    
    def init_new_run(
            self,
            epoch_begin: int,
            step_begin: int,
            log_every_n_steps: int,
            num_train_steps_per_epoch: Optional[int] = None,
            num_val_steps_per_epoch: Optional[int] = None,
            num_test_steps_per_epoch: Optional[int] = None,
        ):
        self._run += 1
        with self._logs as logfile:
            run = logfile.create_group(f"run={self._run}")
            run.create_dataset("epoch_begin", dtype = np.uint32, data = epoch_begin)
            run.create_dataset("step_begin", dtype = np.uint32, data = step_begin)
            run.create_dataset("log_every_n_steps", dtype = np.uint32, data = log_every_n_steps)
        self.num_steps_per_epoch = {
            "train": num_train_steps_per_epoch,
            "val": num_val_steps_per_epoch,
            "test": num_test_steps_per_epoch,
        }
    
    def init_metrics(self, metrics: list[str]):
        with self._logs as logfile:
            if (run := logfile.get(f"run={self._run}")) is None:
                raise TypeError(f"run = {self._run} was not initialized")
            for metric in metrics:
                if (interval := metric.split('_')[-1]) not in ("step", "epoch"):
                    raise ValueError(f"interval must be one of step, epoch, got {interval} in {metric}")
                if (split := metric.split('_')[0]) not in ("train", "val", "test"):
                    raise ValueError(f"split must be one of train, val, test, got {split} in {metric}")
                match interval:
                    case "step":
                        if self.num_steps_per_epoch[split] is None: 
                            return TypeError(f"run = {self._run} was not initialized with num_{split}_steps_per_epoch") 
                        run.create_dataset(metric, shape = (self.num_steps_per_epoch[split],), dtype = np.float32)
                    case "epoch":
                        run.create_dataset(metric, shape = (1,), dtype = np.float32)
    
    def init_confusion_matrix(self, split: str, num_classes: int):
        with self._logs as logfile:
            logfile.create_dataset(f"run={self._run}/{split}_confusion_matrix_epoch", shape = (1, num_classes, num_classes), dtype = np.uint32)
    
    def init_eval_logits(self, split: str, num_classes: Optional[int] = None, top_k: Optional[int] = None):
        with self._logs as logfile:
            if num_classes is not None and top_k is None:
                logfile.create_dataset(f"run={self._run}/{split}_logits_epoch", shape = (1, self.num_steps_per_epoch[split], num_classes), dtype = np.float32)
            elif num_classes is None and top_k is not None:
                logfile.create_dataset(f"run={self._run}/{split}_top_k_logits_epoch", shape = (1, self.num_steps_per_epoch[split], top_k), dtype = np.float32)
                logfile.create_dataset(f"run={self._run}/{split}_top_k_classes_epoch", shape = (1, self.num_steps_per_epoch[split], top_k), dtype = np.uint32)
            else:
                raise ValueError(f"either :num_classes or :top_k must be provided, got num_classes = {num_classes} top_k = {top_k}")
    
    def init_new_epoch(self):
        with self._logs as logfile:
            run = logfile.get(f"run={self._run}")
            for metric in [metric for metric in run.keys() if metric.split('_')[-1] == "step"]:
                split, array = metric.split('_')[0], run.get(metric)
                array.resize((array.shape[0] + self.num_steps_per_epoch[split],))
            for metric in [metric for metric in run.keys() if metric.split('_')[-1] == "epoch"]:
                array = run.get(metric)
                array.resize((array.shape[0]+1, *array.shape[1:]))

    def log_metrics(self, idx: int, metrics: dict[str, Any]):
       with self._logs as logfile:
            for name, value in metrics.items():
                logfile[f"run={self._run}/{name}"][idx] = value
    
    def dump_metric_steps_csv(self):
        pass
    
class ClassificationMetricsLogger(Callback):
    def __init__(self, config: ExperimentConfig, log_every_n_steps: int):
        super().__init__()
        self._config = config
        self._experiment = ExperimentWriter(root = get_experiments_dir(config))
        self._log_every_n_steps = log_every_n_steps
        self._metric_params = self._config

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        match stage:
            case "fit":
                self._dataset = trainer.datamodule.train_dataset
            case "val":
                self._dataset = trainer.datamodule.val_dataset
            case "test":
                self._dataset = trainer.datamodule.test_dataset

        _batch_size = self.config.dataloader_params.batch_size // self.config.dataloader_params.gradient_accumulation
        self.num_train_steps_per_epoch = self._dataset.num_train_samples // _batch_size,
        self.num_val_steps_per_epoch = self._dataset.num_val_samples // _batch_size,
        self.num_test_steps_per_epoch = self._dataset.num_test_samples // _batch_size

        self._experiment.init_new_run(
            epoch_begin=trainer.current_epoch, 
            step_begin=trainer.global_step,
            log_every_n_steps=self._log_every_n_steps,
            num_train_steps_per_epoch=self.num_train_steps_per_epoch,
            num_val_steps_per_epoch=self.num_val_steps_per_epoch,
            num_test_steps_per_epoch=self.num_test_steps_per_epoch,
        ) 
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._experiment.init_metrics([
            "train_loss_step",
            "train_loss_epoch",
            f"train_{self._config.metric}_step",
            f"train_{self._config.metric}_epoch",
        ])
        self.train_step = 0 
        self.train_metric = self._config.get_metric(self._config.metric, pl_module.metric_params)
        self.train_loss_step = np.empty((self.num_train_steps_per_epoch,), np.float32)
        self.train_metric_step = np.empty((self.num_train_steps_per_epoch,), np.float32)
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Mapping[str, Any], batch: Any, batch_idx: int) -> None:
        self.train_loss_step[self.train_step] = outputs["loss"].item()
        self.train_metric_step[self.train_step] = self.train_metric()
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_end(trainer, pl_module)

# class ModelOutputLogger:
# def __init__(self, split: Literal["val", "test"], config, trainer: Trainer, pl_module: LightningModule):
    # match split:
            # case "val":
                # self.dataset = trainer.val_dataloaders.dataset
            # case "test":
                # self.dataset = trainer.test_dataloaders.dataset
            # case _:
                # raise ValueError(f":split must be one of val or test, got {split}")
        # self.split = split
        # self.config = config
        # self.experiments_dir = get_experiments_dir(config)
        # self.output_file_path = get_new_dir(self.experiments_dir, f"{split}_logs") / f"epoch={trainer.current_epoch}_step={trainer.global_step}_outputs.h5"
        # self.output_file_idx = count()
        # self.setup_logfile()

    # def setup_logfile(self):
        # num_eval_samples = len(self.dataset)
        # num_output_classes = self.dataset.num_classes
        # with h5py.File(self.output_file_path, 'w') as f:
            # f.create_dataset("probs", (num_eval_samples, num_output_classes), np.float32)
            # f.create_dataset("labels", (num_eval_samples), np.uint8)
            # f.create_dataset("df_idxs", (num_eval_samples), np.uint8)
    
    # def log(self, outputs: torch.Tensor, batch: tuple):
        # probs = torch.softmax(outputs, 1).detach().cpu().numpy().astype(np.float32)
        # labels = batch[1].cpu().numpy().astype(np.uint8)
        # df_idxs = batch[2].cpu().numpy().astype(np.uint8)
        # with h5py.File(self.output_file_path, "r+") as f:
            # for i, out_idx in zip(range(len(probs)), self.output_file_idx):
                # f["probs"][out_idx] = probs[i]
                # f["labels"][out_idx] = labels[i]
                # f["df_idxs"][out_idx] = df_idxs[i]

    # def log_confusion_matrix(self):
        # confusion_matrix = torchmetrics.functional.confusion_matrix(
            # preds = self.get_probs(),
            # target = self.get_labels(),
            # task = "multiclass" if self.config.dataset.num_classes > 2 else "binary",
            # num_classes = self.config.dataset.num_classes,
        # ).numpy().astype(np.uint8)
        # with h5py.File(self.output_file_path, "r+") as f:
            # f.create_dataset("confm", data = confusion_matrix)

    # def log_metrics_df(self):
        # metrics_df = get_classification_metrics_df(self.get_confusion_matrix(), self.config.dataset.class_names)
        # metrics_df.to_hdf(self.output_file_path, key = "metrics_df", mode = "r+")
    
    # def log_metrics(self):
        # self.log_confusion_matrix()
        # self.log_metrics_df()

    # def get_probs(self) -> torch.Tensor:
        # with h5py.File(self.output_file_path, "r") as f:
            # return torch.tensor(f["probs"][:])
    
    # def get_labels(self) -> torch.Tensor:
        # with h5py.File(self.output_file_path, "r") as f:
            # return torch.tensor(f["labels"][:])

    # def get_df_idxs(self) -> torch.Tensor:
        # with h5py.File(self.output_file_path, "r") as f:
            # return torch.tensor(f["df_idxs"][:])
    
    # def get_confusion_matrix(self) -> NDArray:
        # with h5py.File(self.output_file_path, "r") as f:
            # return f["confm"][:]
    
    # def get_metrics_df(self) -> pd.DataFrame:
        # return pd.read_hdf(self.output_file_path, "metrics_df", "r")
    
    # def get_metrics_dict(self) -> dict:
        # metrics_dict = get_classification_metrics_dict(self.get_metrics_df())
        # return {f"{self.split}/{k}":v for k,v in metrics_dict.items()}

# class ClassificationMetricsLogger(Callback):
    # # store model output logits in memory in a pre-reserved numpy array during eval step
    # # save model outputs and compute confusion matrix and other matrices during eval epoch end 
    # # log everything to CSVLogger and WandbLogger during eval end

    # def __init__(self, config):
        # super().__init__()
        # self.config = config
        # self.experiments_dir = get_experiments_dir(config)
    
    # def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # for logger in trainer.loggers:
            # if isinstance(logger, CSVLogger):
                # self.csv_logger = logger
            # elif isinstance(logger, WandbLogger):
                # self.wandb_logger = logger

        # dataset_df: pd.DataFrame
        # if hasattr(trainer.datamodule, "train_dataset"):
            # dataset_df = trainer.datamodule.train_dataset.df
        # elif hasattr(trainer.datamodule, "val_dataset"):
            # dataset_df = trainer.datamodule.val_dataset.df
        # elif hasattr(trainer.datamodule, "test_dataset"):
            # dataset_df = trainer.datamodule.test_dataset.df
        # else:
            # raise Exception("couldn't load df from datamodule.train/val/test_dataset") 

        # if hasattr(self, "csv_logger"):
            # dataset_df.to_csv(self.experiments_dir/"dataset.csv", index = False)
        # if hasattr(self, "wandb_logger"):
            # self.wandb_logger.log_table("dataset", dataframe = dataset_df)
    
    # def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # self.output_logger = ModelOutputLogger("val", self.config, trainer, pl_module)

    # def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # self.output_logger.log(outputs, batch)
    
    # def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # self.output_logger.log_metrics()
        # pl_module.log_dict(self.output_logger.get_metrics_dict())

    # def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # epoch, step = trainer.current_epoch, trainer.global_step
        # self.logs_dir = get_new_dir(self.experiments_dir / "val_logs")
        # confm_fig = plot_confusion_matrix(self.output_logger.get_confusion_matrix(), self.logs_dir / f"epoch={epoch}_step={step-1}_confm.png")
        # _ = plot_metrics_table(self.output_logger.get_metrics_df(), self.logs_dir / f"epoch={epoch}_step={step-1}_metrics.png")

        # if hasattr(self, "wandb_logger"):
            # self.wandb_logger.log_table(key = "val/metrics", dataframe = self.output_logger.get_metrics_df(), step = step-1)
            # self.wandb_logger.experiment.log({"val/confusion_matrix": confm_fig, "trainer/global_step": step-1})
        # plt.close("all")

    # # def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # self.output_logger = ModelOutputLogger("test", self.config, trainer, pl_module)
    # def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # self.output_logger.log(outputs, batch)
    # def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # self.output_logger.log_metrics()
        # pl_module.log_dict(self.output_logger.get_metrics_dict())
    # def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # if self.wandb_logger is not None:
            # import wandb
            # wandb.log({f"test/confm_epoch={trainer.current_epoch}_step={trainer.global_step}" : wandb.plot.confusion_matrix(
                # probs = self.output_logger.get_probs().numpy(),
                # y_true = self.output_logger.get_labels().numpy(),
                # class_names = self.config.dataset.class_names)
            # })

# class SegmentationMetricsLogger(Callback):
    # pass

def get_csv_logger(config, log_freq: int = 100):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging csv to {experiments_dir}")
    return CSVLogger(
       save_dir = experiments_dir.parent.parent,
       name = experiments_dir.parent.name,
       version = experiments_dir.name,
       flush_logs_every_n_steps = log_freq,
    )

def get_wandb_logger(config):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging wandb to {experiments_dir}")
    return WandbLogger(
        project = experiments_dir.parent.name,
        name = experiments_dir.name,
        save_dir = experiments_dir,
        log_model = "all",
        save_code = False
    )

def get_ckpt_logger(config, save_top_k: int = -1):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging checkpoints to {experiments_dir}")

    min_metrics = ("loss")
    return ModelCheckpoint(
        dirpath = experiments_dir / "ckpts",
        filename = "{epoch}_{step}",
        auto_insert_metric_name = True,
        monitor = f"val/{config.metric}_epoch",
        mode = "min" if (config.metric in min_metrics) else "max",
        save_top_k = save_top_k,
        every_n_epochs = None,
        every_n_train_steps = None,
        train_time_interval = None,
        save_on_train_epoch_end = False,
        enable_version_counter = True
    )

def get_lr_logger(config):
    return LearningRateMonitor(
        logging_interval = "epoch",
        log_momentum = True,
        log_weight_decay = True
    )

def get_classification_logger(config):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging classification metrics to {experiments_dir}")
    return ClassificationMetricsLogger(config)

# def get_segmentation_logger(config):
    # experiments_dir = get_experiments_dir(config)
    # info_logger = get_logger("metrics_logger")
    # info_logger.info(f"logging segmentation metrics to {experiments_dir}")
    # return SegmentationMetricsLogger(config)