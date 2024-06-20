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
from geovision.config.parsers import get_task
from geovision.data.dataset import Dataset
from geovision.logging import get_logger
from geovision.io.local import get_new_dir, get_experiments_dir
from geovision.analysis.viz import plot_confusion_matrix, plot_metrics_table
from .metrics import get_classification_metrics_df, get_classification_metrics_dict

class ExperimentWriter:
    def __init__(self, experiments_dir: Path):
        self.logfile = experiments_dir / "experiment.h5"
    
    def init_run(self, run: int | Literal["new", "last"], log_every_n_steps: int):
        with h5py.File(self.logfile, mode = 'a') as logfile:
            self.run_idxs = sorted(int(key.removeprefix("run_")) for key in logfile.keys())
            self.prev_run_idx = -1 if len(self.run_idxs) == 0 else self.run_idxs[-1]

        if isinstance(run, str):
            if run == "new":
                self.run_idx = self.prev_run_idx + 1
            elif run == "last":
                self.run_idx = self.prev_run_idx
            else:
                raise ValueError(f"invalid :run str, should be either 'new', 'last' or an integer, got {run}")
        elif isinstance(run, int) and run in self.run_idxs:
            self.run_idx = run
        else:
            raise ValueError(f"invalid :run int, should be one of existing runs, from [{self.run_idxs}] or a string, got {run}")

        with h5py.File(self.logfile, mode = 'r+') as logfile:
            run_logs = logfile.create_group(f"run_{self.run_idx}")
            run_logs.create_dataset("epoch_begin", shape = 1, dtype = np.uint32)
            run_logs.create_dataset("step_begin", shape = 1, dtype = np.uint32)
            run_logs.create_dataset("log_every_n_steps", shape = 1, dtype = np.uint32, data = log_every_n_steps)
    
    def update_run_step(self, epoch, step):
        with h5py.File(self.logfile, mode = 'r+') as logfile:
            run_logs = logfile[f"run_{self.run_idx}"]
            run_logs["epoch_begin"][:] = epoch
            run_logs["step_begin"][:] = step 
    
    def init_metric_arrays(self, split: str, metrics: list[str], log_steps_per_epoch: int):
        # print(f"ExperimentWriter: initializing {split} metric array with length = {log_steps_per_epoch}")
        with h5py.File(self.logfile, 'r+') as logfile:
            run_logs = logfile[f"run_{self.run_idx}"]
            if run_logs.get(f"{split}_step_idx") is None:
                run_logs.create_dataset(f"{split}_step_idx", shape = 1, dtype = np.uint32, data = 0)
            if run_logs.get(f"{split}_epoch_idx") is None:
                run_logs.create_dataset(f"{split}_epoch_idx", shape = 1, dtype = np.uint32, data = 0)
            for metric in [f"{split}_{m}" for m in metrics]:
                if run_logs.get(metric) is not None:
                    continue
                suffix = metric.split('_')[-1]
                if suffix == "step":
                    run_logs.create_dataset(
                        name = metric,
                        shape = int(log_steps_per_epoch),
                        dtype = np.float32
                    )
                elif suffix == "epoch":
                    run_logs.create_dataset(
                        name = metric,
                        shape = 1,
                        dtype = np.float32
                    )
                else:
                    raise ValueError(f"metric suffix must be either _step or _epoch, got {suffix}")
    
    def init_confusion_matrix(self, split: str, num_classes: int):
        with h5py.File(self.logfile, mode = 'r+') as logfile:
            run_logs = logfile[f"run_{self.run_idx}"]
            if run_logs.get(f"{split}_confusion_matrix_epoch") is None:
                run_logs.create_dataset(
                    name = f"{split}_confusion_matrix_epoch",
                    shape = (1, num_classes, num_classes),
                    dtype = np.uint32
                )
    
    def init_eval_outputs(self, split: str, samples_per_epoch: int, num_classes: Optional[int] = None, top_k: Optional[int] = None):
        with h5py.File(self.logfile, mode = 'r+') as logfile:
            run_logs = logfile[f"run_{self.run_idx}"]
            if num_classes is not None and top_k is None and run_logs.get(f"{split}_outputs_epoch") is None:
                run_logs.create_dataset(
                    name = f"{split}_outputs_epoch",
                    shape = (1, samples_per_epoch, num_classes),
                    dtype = np.float32
                )
            elif num_classes is None and top_k is not None and run_logs.get(f"{split}_outputs_epoch") is None:
                run_logs.create_dataset(
                    name = f"{split}_outputs_epoch",
                    shape = (1, samples_per_epoch, top_k),
                    dtype = np.float32
                )
                run_logs.create_dataset(
                    name = f"{split}_classes_epoch",
                    shape = (1, samples_per_epoch, top_k),
                    dtype = np.uint32
                )
            elif (num_classes is None and top_k is None) or (num_classes is not None and top_k is not None):
                raise ValueError(f"either :num_classes or :top_k must be passed, got num_classes = {num_classes} top_k = {top_k}")

    def log_metrics(self, split: str, metrics: dict[str, Any]):
        def resize_dataset(file_handle: h5py.File, name: str, scalar: int = 2):
            ds, name_ = file_handle[name], f"{name}_temp"
            ds_ = file_handle.create_dataset(name_, shape = (ds.shape[0]*scalar, *ds.shape[1:]), dtype = ds.dtype)
            # print(f"ExperimentWriter: resizing {name} from {ds.shape} to {ds_.shape}")
            ds_[:len(ds)] = ds[:]
            del file_handle[name]
            file_handle[name] = file_handle[name_]
            del file_handle[name_]

        with h5py.File(self.logfile, mode = 'r+') as logfile:
            step_idx, step_incr = logfile[f"run_{self.run_idx}/{split}_step_idx"], 0
            epoch_idx, epoch_incr = logfile[f"run_{self.run_idx}/{split}_epoch_idx"], 0
            for metric, value in metrics.items():
                metric = f"run_{self.run_idx}/{split}_{metric}" 
                if metric.endswith("_step"): 
                    # print(f"ExperimentWriter: {metric}[{step_idx[0]}] = {value}, buf_len = {len(logfile[metric])}")
                    try:
                        logfile[metric][step_idx[0]] = value
                    except IndexError:
                        # print(f"ExperimentWriter: {metric} needs resizing")
                        resize_dataset(logfile, metric)
                        logfile[metric][step_idx[0]] = value
                    step_incr = 1
                elif metric.endswith("_epoch"):
                    # print(f"ExperimentWriter: {metric}[{epoch_idx[0]}] = {value}, buf_len = {len(logfile[metric])}")
                    try:
                        logfile[metric][epoch_idx[0]] = value
                    except IndexError:
                        # print(f"ExperimentWriter: {metric} needs resizing")
                        resize_dataset(logfile, metric)
                        logfile[metric][epoch_idx[0]] = value
                    epoch_incr = 1
            step_idx[0] += step_incr 
            epoch_idx[0] += epoch_incr
        
    def trim_run(self):
        def trim_dataset(file_handle, name, trim_upto: int):
            ds, name_ = file_handle[name], f"{name}_temp"
            file_handle.create_dataset(name_, data = ds[:trim_upto])
            del file_handle[name]
            file_handle[name] = file_handle[name_]
            del file_handle[name_]

        with h5py.File(self.logfile, mode = 'r+') as logfile:
            for metric in logfile[f"run_{self.run_idx}"].keys():
                split = metric.split('_')[0]
                metric = f"run_{self.run_idx}/{metric}"
                if metric.endswith("_step"):
                    step_idx = logfile[f"run_{self.run_idx}/{split}_step_idx"][0]
                    trim_dataset(logfile, metric, step_idx)
                elif metric.endswith("_epoch"):
                    epoch_idx = logfile[f"run_{self.run_idx}/{split}_epoch_idx"][0]
                    trim_dataset(logfile, metric, epoch_idx)
    
class ClassificationMetricsLogger(Callback):
    def __init__(self, config: ExperimentConfig, log_every_n_steps: int, log_top_k: int = 0):
        super().__init__()
        self.config = config
        self.log_every_n_steps = log_every_n_steps
        if log_top_k < -1:
            raise ValueError(f":log_top_k must be set to -1 for logging entire output, 0 to disable or any top-k (+ve int), got {log_top_k}")
        self.log_top_k = log_top_k

        self.experiment = ExperimentWriter(get_experiments_dir(config))
        self.batch_size = ( 
            config.dataloader_params.batch_size //
            config.dataloader_params.gradient_accumulation
        )
        self.metric_params = {  
            "task": get_task(self.config.dataset),
            "num_classes": self.config.dataset.num_classes,
        }
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.experiment.init_run("new", self.log_every_n_steps)
    
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        self.experiment.trim_run()
    
    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]):
        self.experiment.update_run_step(checkpoint.get("epoch", 0), checkpoint.get("global_step", 0))
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_epoch = 0
        self.train_metric_fn = self.config.get_metric(self.config.metric, self.metric_params)
        if trainer.limit_train_batches != 1.0:
            self.train_steps_per_epoch = int(trainer.limit_train_batches)
        else:
            self.train_steps_per_epoch = int(np.ceil(trainer.datamodule.train_dataset.num_train_samples / self.batch_size))
        self.train_loss_buffer = np.empty(self.train_steps_per_epoch, np.float32)
        self.train_metric_buffer = np.empty(self.train_steps_per_epoch, np.float32)
        self.experiment.init_metric_arrays(
            split = "train",
            log_steps_per_epoch = self.train_steps_per_epoch // self.log_every_n_steps,
            metrics = ["loss_step", "loss_epoch", f"{self.config.metric}_step", f"{self.config.metric}_epoch"]
        )
        
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_step = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Mapping[str, Any], batch: Any, batch_idx: int) -> None:
        if not trainer.validating and not trainer.sanity_checking:
            self.train_loss_buffer[self.train_step] = outputs["loss"].item()
            self.train_metric_buffer[self.train_step] = self.train_metric_fn(outputs["preds"].detach().cpu(), batch[1].detach().cpu()).item()

            if ((self.train_step+1) % self.log_every_n_steps == 0) or (self.train_step == self.train_steps_per_epoch):
                begin, end = self.train_step+1-self.log_every_n_steps, self.train_step+1
                self.experiment.log_metrics(
                    split = "train",
                    metrics = {
                        "loss_step": np.mean(self.train_loss_buffer[begin:end]),
                        f"{self.config.metric}_step": np.mean(self.train_metric_buffer[begin:end]),
                    }
                )
            self.train_step += 1
                
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.validating and not trainer.sanity_checking and self.train_step == self.train_steps_per_epoch:
            self.experiment.log_metrics(
                split = "train",
                metrics = {
                    "loss_epoch": np.mean(self.train_loss_buffer),
                    f"{self.config.metric}_epoch": np.mean(self.train_metric_buffer),
                }
            )
            self.train_epoch += 1
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_epoch = 0
        self.val_metric_fn = self.config.get_metric(self.config.metric, self.metric_params)
        self.val_confm_fn = self.config.get_metric("confusion_matrix", self.metric_params)
        self.num_classes = trainer.datamodule.val_dataset.num_classes
        if trainer.limit_val_batches != 1.0:
            self.val_steps_per_epoch = int(trainer.limit_val_batches)
            self.val_samples_per_epoch = int(trainer.limit_val_batches) * self.batch_size
        else:
            self.val_samples_per_epoch = trainer.datamodule.val_dataset.num_val_samples
            self.val_steps_per_epoch = int(np.ceil(trainer.datamodule.val_dataset.num_val_samples / self.batch_size))
        self.val_loss_buffer = np.empty(self.val_steps_per_epoch, np.float32)
        self.val_metric_buffer = np.empty(self.val_steps_per_epoch, np.float32)
        self.experiment.init_metric_arrays(
            split = "val",
            log_steps_per_epoch = self.val_steps_per_epoch // self.log_every_n_steps,
            metrics = ["loss_step", "loss_epoch", f"{self.config.metric}_step", f"{self.config.metric}_epoch"]
        )
        self.experiment.init_confusion_matrix(
            split = "val",
            num_classes = self.num_classes
        )
        if self.log_top_k:
            if self.log_top_k == -1 or self.log_top_k >= self.num_classes:
                self.experiment.init_eval_outputs("val", self.val_samples_per_epoch, num_classes = self.num_classes)
                self.val_outputs_buffer = np.empty((self.val_samples_per_epoch, self.num_classes))
            else:
                self.experiment.init_eval_outputs("val", self.val_samples_per_epoch, top_k = self.log_top_k)
                self.val_outputs_buffer = np.empty((self.val_samples_per_epoch, self.log_top_k))
                self.val_classes_buffer = np.empty((self.val_samples_per_epoch, self.log_top_k))
                
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_step = 0

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Mapping[str, Any], batch: Any, batch_idx: int) -> None:
        if not trainer.sanity_checking:
            loss, preds, labels = outputs["loss"].item(), outputs["preds"].detach().cpu(), batch[1].detach().cpu()

            self.val_loss_buffer[self.val_step] = loss 
            self.val_metric_buffer[self.val_step] = self.val_metric_fn(preds, labels).item()
            self.val_confm_fn.update(preds, labels)

            if self.log_top_k:
                start, end = self.val_step * len(preds), min((self.val_step+1)*len(preds), self.val_samples_per_epoch) 
                if self.log_top_k == -1 or self.log_top_k >= self.num_classes:
                    self.val_outputs_buffer[start:end] = preds.softmax(1).numpy()
                else:
                    topk = torch.topk(preds.softmax(1), self.log_top_k)
                    self.val_outputs_buffer[start:end] = topk.values.numpy()
                    self.val_classes_buffer[start:end] = topk.indices.numpy()
        
            if ((self.val_step+1) % self.log_every_n_steps == 0) or (self.val_step == self.val_steps_per_epoch):
                begin, end = self.val_step+1-self.log_every_n_steps, self.val_step+1
                self.experiment.log_metrics(
                    split = "val",
                    metrics = {
                        "loss_step": np.mean(self.val_loss_buffer[begin:end]),
                        f"{self.config.metric}_step": np.mean(self.val_metric_buffer[begin:end]),
                    }
                )
            self.val_step += 1
                
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking and self.val_step == self.val_steps_per_epoch:
            epoch_metrics = dict() 
            epoch_metrics["loss_epoch"] = np.mean(self.val_loss_buffer),
            epoch_metrics[f"{self.config.metric}_epoch"] = np.mean(self.val_metric_buffer),
            epoch_metrics["confusion_matrix_epoch"] = self.val_confm_fn.compute().numpy()
            if self.log_top_k:
                if self.log_top_k == -1 or self.log_top_k >= self.num_classes:
                    epoch_metrics["outputs_epoch"] = self.val_outputs_buffer
                else:
                    epoch_metrics["outputs_epoch"] = self.val_outputs_buffer
                    epoch_metrics["classes_epoch"] = self.val_classes_buffer
 
            self.experiment.log_metrics("val", epoch_metrics)
            self.val_confm_fn.reset()
            self.val_epoch += 1

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

def get_classification_logger(config, log_every_n_steps: int = 50, log_top_k: int = 0):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging classification metrics to {experiments_dir}")
    return ClassificationMetricsLogger(config, log_every_n_steps, log_top_k)

# def get_segmentation_logger(config):
    # experiments_dir = get_experiments_dir(config)
    # info_logger = get_logger("metrics_logger")
    # info_logger.info(f"logging segmentation metrics to {experiments_dir}")
    # return SegmentationMetricsLogger(config)