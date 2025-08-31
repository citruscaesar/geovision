from typing import Any, Mapping, Optional, Literal
from numpy.typing import NDArray

import wandb
import h5py
import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from geovision.data import Dataset
from geovision.experiment.config import ExperimentConfig
from .utils import plot as cfm_plot

import logging
logger = logging.getLogger(__name__)

class HDF5ExperimentWriter:
    def __init__(self, experiments_dir: Path):
        self.logfile = experiments_dir / "experiment.h5"

        with h5py.File(self.logfile, mode="a", libver="latest") as logfile:
            self.run_idxs = sorted(int(key.removeprefix("run=")) for key in logfile.keys())
            self.run_idx = 0 if len(self.run_idxs) == 0 else self.run_idxs[-1] + 1
            logfile.create_group(f"run={self.run_idx}", track_order=True)

    def add_metadata(self, key: str, value: int | list) -> None:
        """add metadata used to interpret metric buffers. :step/epoch begin points to where the ckpt was loaded, :step/epoch interval is the logging 
        interval between iterations. :steps_per_epochs"""
        with h5py.File(self.logfile, mode="r+") as logfile:
            logfile[f"run={self.run_idx}"].attrs.modify(key, value)
    
    def add_metric_buffers(self, metrics: list[str], shape: int | tuple, dtype):
        """add metric arrays as run=run_idx/split_metric_suffix"""

        with h5py.File(self.logfile, mode="r+") as logfile:
            for metric in metrics:
                metric = f"run={self.run_idx}/{metric}"
                if logfile.get(metric) is None:
                    logfile.create_dataset(metric, shape=shape, dtype=dtype)
                    logfile[metric].attrs.modify("idx", 0)

    def log_dict(self, metrics: dict[str, Any]):
        """logs metrics dict to run=run_idx/split_metric_suffix and resizes the dataset array if needed"""

        def resize_buffer(metric: str, idx: int):
            metadata = {k:v for k, v in logfile[metric].attrs.items()}
            while idx >= len(logfile[metric]):
                logfile.create_dataset("temp", shape = (logfile[metric].shape[0]*2, *logfile[metric].shape[1:]), dtype = logfile[metric].dtype)
                logfile["temp"][:len(logfile[metric])] = logfile[metric] # NOTE: if this reads the buffer into memory, this whole op is a waste of time
                del logfile[metric]
                logfile[metric] = logfile["temp"] 
                del logfile["temp"]
            for k,v in metadata.items():
                logfile[metric].attrs.modify(k,v)

        with h5py.File(self.logfile, mode="r+") as logfile:
            logfile.swmr_mode = True
            for metric, value in metrics.items():
                metric = f"run={self.run_idx}/{metric}"
                resize_buffer(metric, logfile[metric].attrs["idx"])
                dataset = logfile[metric]
                dataset[dataset.attrs["idx"]] = value
                dataset.attrs.modify("idx", dataset.attrs["idx"]+1)
                
    def trim_run(self):
        """remove buffer space upto {split}_step_end and {split}_epoch_end for current run to reduce storage footprint. Must be called to end a run"""
        with h5py.File(self.logfile, mode="r+") as logfile:
            for metric in (f"run={self.run_idx}/{m}" for m in logfile[f"run={self.run_idx}"].keys()):
                logfile["temp"] = logfile[metric][:logfile[metric].attrs["idx"]]
                del logfile[metric] 
                logfile[metric] = logfile["temp"]
                del logfile["temp"] 

class ClassificationLogger(Callback):
    def __init__(self, config: ExperimentConfig):
        self.log_every_n_steps: int = config.log_params["log_every_n_steps"]
        self.log_every_n_epochs: int = config.log_params["log_every_n_epochs"]
        # self.log_model_outputs: int = config.log_params["log_model_outputs"]  # -1 means log all, 0 means none, int < num_classes means top_k
        self.log_to_h5: bool = config.log_params["log_to_h5"]
        self.log_to_wandb: bool = config.log_params["log_to_wandb"]
        self.log_to_csv: bool = config.log_params["log_to_csv"]

        if self.log_to_wandb:
            self.wandb_init_params: dict = config.wandb_init_params

        self.experiments_dir: Path = config.experiments_dir 
        self.batch_size: int = config.dataloader_config.batch_size // config.dataloader_config.gradient_accumulation
        self.learning_rate: float = config.optimizer_params["lr"]
        self.dataset: Dataset = config.dataset_constructor
        self.monitor_metric_name: str = config.metric_name

        self.metrics = torchmetrics.MetricCollection({
            "accuracy": config.get_metric("Accuracy", {"sync_on_compute": True}),
        })
        
    def calculate_steps(self, trainer: Trainer):
        if trainer.limit_train_batches != 1.0:
            self.train_steps_per_epoch = int(trainer.limit_train_batches)
            self.train_samples_per_epoch = int(trainer.limit_train_batches) * self.batch_size
        else:
            self.train_samples_per_epoch = trainer.datamodule.val_dataset.num_train_samples
            self.train_steps_per_epoch = int(np.ceil(trainer.datamodule.val_dataset.num_train_samples / self.batch_size))

        if trainer.limit_val_batches != 1.0:
            self.val_steps_per_epoch = int(trainer.limit_val_batches)
            self.val_samples_per_epoch = int(trainer.limit_val_batches) * self.batch_size
        else:
            self.val_samples_per_epoch = trainer.datamodule.val_dataset.num_val_samples
            self.val_steps_per_epoch = int(np.ceil(trainer.datamodule.val_dataset.num_val_samples / self.batch_size))

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.is_global_zero:
            self.calculate_steps(trainer)
            if self.log_to_h5:
                self.h5_run = HDF5ExperimentWriter(self.experiments_dir)
                self.h5_run.add_metadata("step_begin", 0)
                self.h5_run.add_metadata("step_interval", self.log_every_n_steps)
                self.h5_run.add_metadata("epoch_begin", 0)
                self.h5_run.add_metadata("epoch_interval", self.log_every_n_epochs)
                self.h5_run.add_metadata("num_train_steps_per_epoch", self.train_steps_per_epoch)
                self.h5_run.add_metadata("num_val_steps_per_epoch", self.val_steps_per_epoch)
                self.h5_run.add_metadata("class_names", self.dataset.class_names)
                self.h5_run.add_metadata("monitored_metric", self.monitor_metric_name)
            if self.log_to_wandb: 
                self.wandb_run = wandb.init(**self.wandb_init_params)

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]):
        if trainer.is_global_zero:
            self.epoch_begin = checkpoint.get("epoch", 0)
            self.step_begin = checkpoint.get("global_step", 0)
            if self.log_to_h5:
                self.h5_run.add_metadata("step_begin", self.step_begin)
                self.h5_run.add_metadata("epoch_begin", self.epoch_begin)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_epoch = 0
        self.train_loss_buffer = np.empty(self.train_steps_per_epoch, np.float32)
        self.train_metrics_step = self.metrics.clone(prefix = "train_", postfix = "_step")
        self.train_metrics_epoch = self.metrics.clone(prefix = "train_", postfix = "_epoch")
        self.train_confusion_matrix = self.confusion_matrix.clone()
        if trainer.is_global_zero and self.log_to_h5:
            self.h5_run.add_metric_buffers(
                metrics = list(self.train_metrics_step.keys()) + ["train_loss_step", "train_lr_step"], 
                shape = self.train_steps_per_epoch // self.log_every_n_steps, 
                dtype = np.float32
            )
            self.h5_run.add_metric_buffers(
                metrics = list(self.train_metrics_epoch.keys()) + ["train_loss_epoch", "train_lr_epoch"], 
                shape = 1, 
                dtype = np.float32
            )
            self.h5_run.add_metric_buffers(
                metrics = ["train_confusion_matrix_epoch"],
                shape = (1, self.dataset.num_classes, self.dataset.num_classes),
                dtype = np.uint64
            )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_step = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if not trainer.validating and not trainer.sanity_checking:
            preds, labels, loss = outputs["preds"].detach().cpu(), batch[1].detach().cpu(), outputs["loss"]

            if trainer.world_size > 1:
                local_loss = torch.tensor(loss, device = pl_module.device)
                synced_loss = trainer.strategy.reduce(local_loss, "mean")
                loss = synced_loss.item()
            else:
                loss = loss.item()

            # print("labels", type(labels), labels.dim(), labels.shape)
            if labels.dim() > 1:
                labels = labels.argmax(1)
            self.train_loss_buffer[self.train_step] = loss
            self.train_metrics_step.update(preds, labels)
            self.train_metrics_epoch.update(preds, labels)
            self.train_confusion_matrix.update(preds, labels)

            # log at every :log_every_n_step^th step or at the last step
            if trainer.is_global_zero:
                if ((self.train_step + 1) % self.log_every_n_steps == 0) or (self.train_step == self.train_steps_per_epoch):
                    begin, end = self.train_step + 1 - self.log_every_n_steps, self.train_step + 1
                    metrics_dict = self.train_metrics_step.compute()
                    metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"train_loss_step": np.mean(self.train_loss_buffer[begin:end])}

                    # NOTE: there should always be one scheduler, combining warmup and regular scheduler as sequential_lr with warmup steps
                    scheduler = pl_module.lr_schedulers()
                    if scheduler is not None:
                        if isinstance(scheduler, list):
                            scheduler = scheduler[-1]
                        # NOTE: get_last_lr(): (...) -> [lr]
                        self.learning_rate = scheduler.get_last_lr()[0]
                    else:
                        # TODO: get lr from optmizers instead ? print(pl_module.optimizers())
                        self.learning_rate = np.nan

                    # assert isinstance(self.learning_rate, float), f"logging error, expected :learning_rate to be of type float, got {type(self.learning_rate)}"
                    metrics_dict = metrics_dict | {"train_lr_step": self.learning_rate}

                    if self.log_to_h5:
                        self.h5_run.log_dict(metrics_dict)
                    if self.log_to_csv:
                        pl_module.log_dict(metrics_dict, on_step=True, on_epoch=False)
                    if self.log_to_wandb:
                        self.wandb_run.log(metrics_dict | {"trainer_step": trainer.global_step})
                    self.train_metrics_step.reset()
                self.train_step += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.is_global_zero:
            if not trainer.validating and not trainer.sanity_checking and self.train_step == self.train_steps_per_epoch:
                # print(f"logging epoch at {self.train_epoch}")
                metrics_dict = self.train_metrics_epoch.compute()
                metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"train_loss_epoch": np.mean(self.train_loss_buffer)}

                scheduler = pl_module.lr_schedulers()
                if scheduler is not None:
                    if isinstance(scheduler, list):
                        scheduler = scheduler[-1]
                    self.learning_rate = scheduler.get_last_lr()[0]
                else:
                    self.learning_rate = np.nan

                metrics_dict = metrics_dict | {"train_lr_step": self.learning_rate}

                confusion_matrix = self.train_confusion_matrix.compute().numpy()
                if self.log_to_h5:
                    self.h5_run.log_dict(metrics_dict | {"train_confusion_matrix_epoch": confusion_matrix})
                if self.log_to_csv:
                    pl_module.log_dict(metrics_dict, on_step=False, on_epoch=True)
                if self.log_to_wandb:
                    self.wandb_run.log(metrics_dict | {
                        "train_confusion_matrix_epoch": get_confusion_matrix_plot(confusion_matrix, self.dataset.class_names), 
                        "trainer_step": trainer.global_step, 
                        "trainer_epoch": trainer.current_epoch
                    })
                self.train_metrics_epoch.reset()
                self.train_confusion_matrix.reset()
                self.train_epoch += 1

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_epoch = 0
        self.val_loss_buffer = np.empty(self.val_steps_per_epoch, np.float32)
        self.val_metrics_step = self.metrics.clone(prefix = "val_", postfix = "_step")
        self.val_metrics_epoch = self.metrics.clone(prefix = "val_", postfix = "_epoch")
        self.val_confusion_matrix = self.confusion_matrix.clone()
        if self.log_to_h5:
            self.h5_run.add_metric_buffers(
                metrics = list(self.val_metrics_step.keys()) + ["val_loss_step"], 
                shape = self.val_steps_per_epoch // self.log_every_n_steps,
                dtype = np.float32
            )
            self.h5_run.add_metric_buffers(
                metrics = list(self.val_metrics_epoch.keys()) + ["val_loss_epoch"],
                shape = 1,
                dtype = np.float32
            )
            self.h5_run.add_metric_buffers(
                metrics = ["val_confusion_matrix_epoch"],
                shape = (1, self.dataset.num_classes, self.dataset.num_classes),
                dtype = np.int64
            )

    # def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # self.val_step = 0

    # def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # if trainer.is_global_zero and not trainer.sanity_checking:
            # preds, labels, loss = outputs["preds"].detach().cpu(), batch[1].detach().cpu(), outputs["loss"],
            # if trainer.world_size > 1:
                # local_loss = torch.tensor(loss, device = pl_module.device)
                # synced_loss = trainer.strategy.reduce(local_loss, "mean")
                # loss = synced_loss.item()
            # else:
                # loss = loss.item()

            # if labels.dim() > 1:
                # labels = labels.argmax(1)
            # self.val_loss_buffer[self.val_step] = loss
            # self.val_metrics_step.update(preds, labels)
            # self.val_metrics_epoch.update(preds, labels)
            # self.val_confusion_matrix.update(preds, labels)

            # if ((self.val_step + 1) % self.log_every_n_steps == 0) or (self.val_step == self.val_steps_per_epoch):
                # begin, end = self.val_step + 1 - self.log_every_n_steps, self.val_step + 1
                # metrics_dict = self.val_metrics_step.compute()
                # metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"val_loss_step": np.mean(self.val_loss_buffer[begin:end])}
                # if self.log_to_h5:
                    # self.h5_run.log_dict(metrics_dict)
                # if self.log_to_csv:
                    # pl_module.log_dict(metrics_dict, on_step=True, on_epoch=False)
                # if self.log_to_wandb:
                    # self.wandb_run.log(metrics_dict | {"trainer_step": trainer.global_step + self.val_step})
                # self.val_metrics_step.reset()
            # self.val_step += 1

    # def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # if trainer.is_global_zero:
            # if not trainer.sanity_checking and self.val_step == self.val_steps_per_epoch:
                # metrics_dict = self.val_metrics_epoch.compute()
                # metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"val_loss_epoch": np.mean(self.val_loss_buffer)}
                # confusion_matrix = self.val_confusion_matrix.compute().numpy()
                # if self.log_to_csv:
                    # pl_module.log_dict(metrics_dict, on_step=False, on_epoch=True)
                # if self.log_to_h5:
                    # self.h5_run.log_dict(metrics_dict | {"val_confusion_matrix_epoch": confusion_matrix})
                # if self.log_to_wandb:
                    # self.wandb_run.log(metrics_dict | {
                        # "val_confusion_matrix_epoch": get_confusion_matrix_plot(confusion_matrix, self.dataset.class_names), 
                        # "trainer_step": trainer.global_step, 
                        # "trainer_epoch": trainer.current_epoch
                    # })
                # self.val_metrics_epoch.reset()
                # self.val_confusion_matrix.reset()
                # self.val_epoch += 1

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.is_global_zero:
            if self.log_to_h5:
                self.h5_run.trim_run()
            if self.log_to_wandb:
                self.wandb_run.finish()

def get_confusion_matrix_plot(mat: NDArray, class_names: tuple[str, ...]):
    fig, ax = plt.subplots(1, 1, figsize = (10,8), layout = "constrained")
    cfm_plot(ax, mat, class_names)
    return fig

def get_csv_logger(config):
    return CSVLogger(
        save_dir=config.experiments_dir.parent.parent,
        name=config.experiments_dir.parent.name,
        version=config.experiments_dir.name,
        flush_logs_every_n_steps=config.log_params["log_every_n_steps"],
    )

def get_wandb_logger(config: ExperimentConfig):
    return WandbLogger(**config.wandb_logger_params)

def get_ckpt_logger(config):
    # min_metrics = ("loss",)
    return ModelCheckpoint(
        dirpath=config.experiments_dir / "ckpts",
        filename="{epoch}_{step}",
        auto_insert_metric_name=True,
        # monitor=f"val_{config.metric_name}_epoch",
        # mode="min" if (config.metric_name in min_metrics) else "max",
        save_top_k=config.log_params["log_models"],
        every_n_epochs=None,
        every_n_train_steps=None,
        train_time_interval=None,
        save_on_train_epoch_end=False,
        enable_version_counter=True,
    )

def get_classification_logger(config):
    logger.info(f"logging classification metrics to {config.experiments_dir}")
    return ClassificationLogger(config)
