from typing import Any, Mapping, Optional, Literal
from numpy.typing import NDArray

import wandb
import h5py
import torch
import torchmetrics
import numpy as np
from pathlib import Path
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from geovision.data.dataset import Dataset
from geovision.config.experiment_config import ExperimentConfig
from geovision.analysis.viz import get_confusion_matrix_plot
from geovision.io.local import get_experiments_dir

import logging
logger = logging.getLogger(__name__)

class HDF5ExperimentWriter:
    def __init__(self, experiments_dir: Path):
        self.logfile = experiments_dir / "experiment.h5"

        with h5py.File(self.logfile, mode="a") as logfile:
            self.run_idxs = sorted(int(key.removeprefix("run=")) for key in logfile.keys())
            self.run_idx = 0 if len(self.run_idxs) == 0 else self.run_idxs[-1] + 1
            logfile.create_group(f"run={self.run_idx}")

    def add_metadata(self, step_begin: int = 0, step_interval: int = 1, epoch_begin: int = 0, epoch_interval: int = 1) -> None:
        """add metadata used to interpret metric buffers. :step/epoch begin points to where the ckpt was loaded, :step/epoch interval is the logging interval between iterations"""

        with h5py.File(self.logfile, mode="r+") as logfile:
            run_logs = logfile[f"run={self.run_idx}"]
            run_logs.create_dataset("step_begin", shape=1, dtype=np.uint32, data=step_begin)
            run_logs.create_dataset("epoch_begin", shape=1, dtype=np.uint32, data=epoch_begin)
            run_logs.create_dataset("step_interval", shape=1, dtype=np.uint32, data=step_interval)
            run_logs.create_dataset("epoch_interval", shape=1, dtype=np.uint32, data=epoch_interval)

    def add_metric_buffers(self, metrics: list[str], split: str, suffix: str, buffer_size: int) -> None:
        """add metric arrays as run=run_idx/split_metric_suffix"""

        with h5py.File(self.logfile, mode="r+") as logfile:
            run_logs = logfile[f"run={self.run_idx}"]
            if run_logs.get(f"{split}_{suffix}_end") is None:
                run_logs.create_dataset(f"{split}_{suffix}_end", shape=1, dtype=np.uint32, data=0)
            for metric in metrics:
                if run_logs.get(f"{split}_{metric}_{suffix}") is None:
                    run_logs.create_dataset(f"{split}_{metric}_{suffix}", buffer_size, np.float32)

    def add_confusion_matrix(self, num_classes: int, split: str):
        """add confusion matrix arrays as run=run_idx/split_confusion_matrix_epoch"""

        with h5py.File(self.logfile, mode="r+") as logfile:
            run_logs = logfile[f"run={self.run_idx}"]
            if run_logs.get(f"{split}_confusion_matrix_epoch") is None:
                run_logs.create_dataset(f"{split}_confusion_matrix_epoch", shape=(1, num_classes, num_classes), dtype=np.uint32)

    def add_model_outputs(self, num_samples: int, num_classes: int, is_top_k: bool, split: str):
        """add arrays to store model outputs, beware of large storage footprints. set :is_top_k to True to store class labels alongside outputs"""

        with h5py.File(self.logfile, mode="r+") as logfile:
            run_logs = logfile[f"run={self.run_idx}"]
            if run_logs.get(f"{split}_outputs_epoch") is None:
                run_logs.create_dataset(f"{split}_outputs_epoch", shape=(1, num_samples, num_classes), dtype=np.float32)
            if is_top_k and run_logs.get(f"{split}_classes_epoch") is None:
                run_logs.create_dataset(f"{split}_classes_epoch", shape=(1, num_samples, num_classes), dtype=np.uint32)

    def resize(self, file_handle: h5py.File, name: str, idx: int, scalar: int = 2):
        """internal method, exponentially resizes array at :scalar rate until idx fits inside it, usually only runs once"""

        # print(f"rimentWriter: attempting to resize {name} @ [{idx}], buf_len = {len(file_handle[name])}")
        while idx >= len(file_handle[name]):
            ds, name_ = file_handle[name], f"{name}_temp"
            # print(f"rimentWriter: resizing {name} from {ds.shape} to {ds_.shape}")
            ds_ = file_handle.create_dataset(
                name_, shape=(ds.shape[0] * scalar, *ds.shape[1:]), dtype=ds.dtype
            )
            ds_[: len(ds)] = ds[:]
            del file_handle[name]
            file_handle[name] = file_handle[name_]
            del file_handle[name_]

    def log_dict(self, metrics: dict[str, Any], split: Literal["train", "val", "test"], suffix: Literal["step", "epoch"]):
        """logs metrics dict to run=run_idx/split_metric_suffix and resizes the dataset array if needed"""

        with h5py.File(self.logfile, mode="r+") as logfile:
            insert_at = logfile[f"run={self.run_idx}/{split}_{suffix}_end"]
            for k, v in metrics.items():
                metric = f"run={self.run_idx}/{k}"
                self.resize(logfile, metric, insert_at[0])
                logfile[metric][insert_at[0]] = v
            insert_at[0] += 1

    def log_outputs(self, split: str, start: int, end: int, outputs: NDArray, classes: Optional[NDArray] = None):
        """logs model outputs to run=run_idx/split_outputs_epoch, and optionally logs class idxs in case of top-k outputs"""

        def update_model_outputs(file_handle: h5py.File, name: str, idx: int, buffer: NDArray):
            self.resize(file_handle, name, idx)
            file_handle[name][idx, start:end] = buffer

        with h5py.File(self.logfile, mode="r+") as logfile:
            insert_at = logfile[f"run={self.run_idx}/{split}_epoch_end"]
            update_model_outputs(logfile, f"run={self.run_idx}/{split}_outputs_epoch", insert_at[0], outputs)
            if classes is not None:
                update_model_outputs(logfile, f"run={self.run_idx}/{split}_classes_epoch", insert_at[0], classes)

    def trim_run(self):
        """remove buffer space upto {split}_step_end and {split}_epoch_end for current run to reduce storage footprint"""

        def trim_dataset(file_handle: h5py.File, name: str, trim_upto: int):
            # logger.info(f"ExperimentIO: trimming {name} upto [{trim_upto}]")
            ds, name_ = file_handle[name], f"{name}_temp"
            file_handle.create_dataset(name_, data=ds[:trim_upto])
            del file_handle[name]
            file_handle[name] = file_handle[name_]
            del file_handle[name_]

        with h5py.File(self.logfile, mode="r+") as logfile:
            for split in ("train", "val", "test"):
                step_end = logfile.get(f"run={self.run_idx}/{split}_step_end")
                epoch_end = logfile.get(f"run={self.run_idx}/{split}_epoch_end")
                if step_end is None or epoch_end is None:
                    continue
                for metric in [
                    f"run={self.run_idx}/{m}"
                    for m in logfile[f"run={self.run_idx}"].keys()
                    if m.startswith(split)
                ]:
                    if metric.endswith("_step"):
                        trim_dataset(logfile, metric, trim_upto=step_end[0])
                    elif metric.endswith("_epoch"):
                        trim_dataset(logfile, metric, trim_upto=epoch_end[0])
                del step_end
                del epoch_end


class ClassificationLogger(Callback):
    def __init__(self, config: ExperimentConfig):
        self.log_every_n_steps: int = config.log_params["log_every_n_steps"]
        self.log_every_n_epochs: int = config.log_params["log_every_n_epochs"]
        self.log_model_outputs: int = config.log_params["log_model_outputs"]  # -1 means log all, 0 means none, int < num_classes means top_k
        self.log_to_h5: bool = config.log_params["log_to_h5"]
        self.log_to_wandb: bool = config.log_params["log_to_wandb"]
        self.log_to_csv: bool = config.log_params["log_to_csv"]

        if self.log_to_wandb:
            self.wandb_init_params: dict = config.get_wandb_init_params()

        self.dataset: Dataset = config.dataset_
        self.experiments_dir: Path = get_experiments_dir(config)
        self.batch_size: int = config.dataloader_params.batch_size // config.dataloader_params.gradient_accumulation

        self.metrics = torchmetrics.MetricCollection({
            "precision": config.get_metric("precision"),
            "recall": config.get_metric("recall"),
            "f1": config.get_metric("f1"),
            "iou": config.get_metric("iou"),
            "accuracy": config.get_metric("accuracy"),
            "accuracy_top5": config.get_metric("accuracy", {"top_k": 5}),
        })
        if config.metric not in self.metrics.keys():
            self.metrics.add_metrics({config.metric: config.get_metric(config.metric)})
        self.confusion_matrix = config.get_metric("confusion_matrix")

    def calculate_steps(self, trainer: Trainer):
        if trainer.limit_train_batches != 1.0:
            self.train_steps_per_epoch = int(trainer.limit_train_batches)
            self.train_samples_per_epoch = int(trainer.limit_train_batches) * self.batch_size
        else:
            self.train_samples_per_epoch = trainer.datamodule.train_dataset.num_train_samples
            self.train_steps_per_epoch = int(np.ceil(trainer.datamodule.train_dataset.num_train_samples / self.batch_size))

        if trainer.limit_val_batches != 1.0:
            self.val_steps_per_epoch = int(trainer.limit_val_batches)
            self.val_samples_per_epoch = int(trainer.limit_val_batches) * self.batch_size
        else:
            self.val_samples_per_epoch = trainer.datamodule.val_dataset.num_val_samples
            self.val_steps_per_epoch = int(np.ceil(trainer.datamodule.val_dataset.num_val_samples / self.batch_size))

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.calculate_steps(trainer)
        if self.log_to_h5:
            self.h5_run = HDF5ExperimentWriter(self.experiments_dir)
        if self.log_to_wandb:
            self.wandb_run = wandb.init(**self.wandb_init_params)

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]):
        self.epoch_begin = checkpoint.get("epoch", 0)
        self.step_begin = checkpoint.get("global_step", 0)
        if self.log_to_h5:
            self.h5_run.add_metadata(
                step_begin = self.step_begin,
                step_interval = self.log_every_n_steps,
                epoch_begin = self.epoch_begin,
                epoch_interval = self.log_every_n_epochs,
            )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_epoch = 0
        self.train_loss_buffer = np.empty(self.train_steps_per_epoch, np.float32)
        self.train_metrics_step = self.metrics.clone(prefix = "train_", postfix = "_step")
        self.train_metrics_epoch = self.metrics.clone(prefix = "train_", postfix = "_epoch")
        self.train_confusion_matrix = self.confusion_matrix.clone(prefix = "train_", postfix = "_epoch")
        self.h5_run.add_metric_buffers(metrics = list(self.train_metrics_step.keys()) + ["train_loss_step"], split = "train", suffix = "step", buffer_size = self.train_steps_per_epoch)
        self.h5_run.add_metric_buffers(metrics = list(self.train_metrics_epoch.keys()) + ["train_loss_epoch"], split = "train", suffix = "epoch", buffer_size = 1)
        self.h5_run.add_confusion_matrix(num_classes = self.dataset.num_classes, split = "train")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_step = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if not trainer.validating and not trainer.sanity_checking:
            preds, labels, loss = outputs["preds"].detach().cpu(), batch[1].detach().cpu(), outputs["loss"].item()
            self.train_loss_buffer[self.train_step] = loss
            self.train_metrics_step.update(preds, labels)
            self.train_metrics_epoch.update(preds, labels)
            self.train_confusion_matrix.update(preds, labels)

            # log at every :log_every_n_step^th step or at the last step
            if ((self.train_step + 1) % self.log_every_n_steps == 0) or (self.train_step == self.train_steps_per_epoch):
                begin, end = self.train_step + 1 - self.log_every_n_steps, self.train_step + 1
                metrics_dict = self.train_metrics_step.compute()
                metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"train_loss_step": np.mean(self.train_loss_buffer[begin:end])}
                if self.log_to_h5:
                    self.h5_run.log_dict(metrics_dict, "train", "step")
                if self.log_to_csv:
                    pl_module.log_dict(metrics_dict, on_step=False, on_epoch=False)
                if self.log_to_wandb:
                    self.wandb_run.log(metrics_dict | {"trainer_step": trainer.global_step})
                self.train_metrics_step.reset()
            self.train_step += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.validating and not trainer.sanity_checking and self.train_step == self.train_steps_per_epoch:
            metrics_dict = self.train_metrics_epoch.compute()
            metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"train_loss_epoch": np.mean(self.train_loss_buffer)}
            confusion_matrix = self.train_confusion_matrix.compute().numpy()
            if self.log_to_csv:
                pl_module.log_dict(metrics_dict, on_step=False, on_epoch=False)
            if self.log_to_h5:
                self.h5_run.log_dict(metrics_dict | {"train_confusion_matrix_epoch": confusion_matrix}, "train", "epoch")
            if self.log_to_wandb:
                self.wandb_run.log(metrics_dict | {"train_confusion_matrix_epoch": get_confusion_matrix_plot(confusion_matrix, self.dataset.class_names), "trainer_step": trainer.global_step, "trainer_epoch": trainer.current_epoch})
            self.train_metrics_epoch.reset()
            self.train_confusion_matrix.reset()
            self.train_epoch += 1

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_epoch = 0
        self.val_loss_buffer = np.empty(self.val_steps_per_epoch, np.float32)
        self.val_metrics_step = self.metrics.clone(prefix = "val_", suffix = "_step")
        self.val_metrics_epoch = self.metrics.clone(prefix = "val_", suffix = "_epoch")
        self.val_confusion_matrix = self.confusion_matrix.clone(prefix = "val_", suffix = "_epoch")
        self.h5_run.add_metric_buffers(metrics = list(self.metrics_dict.keys()) + ["loss"], split = "val", suffix = "step", buffer_size = self.val_steps_per_epoch)
        self.h5_run.add_metric_buffers(metrics = list(self.metrics_dict.keys()) + ["loss"], split = "val", suffix = "epoch", buffer_size = 1)
        self.h5_run.add_confusion_matrix(self.dataset.num_classes, "val")

        # log at every :log_every_n_step^th step or at the last step
        if self.log_to_h5 and self.log_model_outputs != 0:
            if self.log_model_outputs == -1 or self.log_model_outputs >= self.dataset.num_classes:
                self.h5_run.add_model_outputs(self.val_samples_per_epoch, self.dataset.num_classes, False, "val")
            else:
                self.h5_run.add_model_outputs(self.val_samples_per_epoch, self.log_model_outputs, True, "val")

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_step = 0

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not trainer.sanity_checking:
            preds, labels, loss = outputs["preds"].detach().cpu(), batch[1].detach().cpu(), outputs["loss"].item(),
            self.val_loss_buffer[self.val_step] = loss
            self.val_metrics_step.update(preds, labels)
            self.val_metrics_epoch.update(preds, labels)
            self.val_confusion_matric.update(preds, labels)

            if ((self.val_step + 1) % self.log_every_n_steps == 0) or (self.val_step == self.val_steps_per_epoch):
                begin, end = self.val_step + 1 - self.log_every_n_steps, self.val_step + 1
                metrics_dict = self.val_metrics_step.compute()
                metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"val_loss_step": np.mean(self.val_loss_buffer[begin:end])}
                if self.log_to_h5:
                    self.h5_run.log_dict(metrics_dict, "val", "step")
                if self.log_to_csv:
                    pl_module.log_dict(metrics_dict, on_step=False, on_epoch=False)
                if self.log_to_wandb:
                    self.wandb_run.log(metrics_dict | {"trainer_step": trainer.global_step + self.val_step})
                self.val_metrics_step.reset()
            self.val_step += 1

            if self.log_to_h5 and self.log_model_outputs != 0:
                start, end = self.val_step * len(preds), min((self.val_step + 1) * len(preds), self.val_samples_per_epoch)
                if self.log_top_k_preds == -1 or self.log_top_k_preds >= self.dataset.num_classes:
                    self.h5_run.log_outputs("val", start, end, preds.numpy())
                else:
                    topk = torch.topk(preds, self.log_top_k_preds)
                    self.h5_run.log_outputs("val", start, end, topk.values.numpy(), topk.indices.numpy())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking and self.val_step == self.val_steps_per_epoch:
            metrics_dict = self.val_metrics_epoch.compute()
            metrics_dict = {k: metrics_dict[k].item() for k in metrics_dict.keys()} | {"val_loss_epoch": np.mean(self.val_loss_buffer)}
            confusion_matrix = self.val_confusion_matrix.compute().numpy()
            if self.log_to_csv:
                pl_module.log_dict(metrics_dict)
            if self.log_to_h5:
                self.h5_run.log_dict(metrics_dict | {"val_confusion_matrix_epoch": confusion_matrix}, "val", "epoch")
            if self.log_to_wandb:
                self.wandb_run.log(metrics_dict | {"val_confusion_matrix_epoch": get_confusion_matrix_plot(confusion_matrix, self.dataset.class_names), "trainer_step": trainer.global_step, "trainer_epoch": trainer.current_epoch})
            self.val_metrics_epoch.reset()
            self.val_confusion_matrix.reset()
            self.val_epoch += 1

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.log_to_h5:
            self.h5_run.trim_run()
        if self.log_to_wandb:
            self.wandb_run.finish()

def get_csv_logger(config):
    logger.info(f"logging metrics.csv to {config.experiments_dir}")
    return CSVLogger(
        save_dir=config.experiments_dir.parent.parent,
        name=config.experiments_dir.parent.name,
        version=config.experiments_dir.name,
        flush_logs_every_n_steps=100,
    )

def get_wandb_logger(config):
    logger.info(f"logging wandb to {config.experiments_dir}")
    return WandbLogger(
        project=config.experiments_dir.parent.name,
        name=config.experiments_dir.name,
        save_dir=config.experiments_dir,
        log_model="all",
        save_code=False,
    )

def get_ckpt_logger(config):
    logger.info(f"logging ckpts to {config.experiments_dir}")

    min_metrics = "loss"
    return ModelCheckpoint(
        dirpath=config.experiments_dir / "ckpts",
        filename="{epoch}_{step}",
        auto_insert_metric_name=True,
        monitor=f"val_{config.metric}_epoch",
        mode="min" if (config.metric in min_metrics) else "max",
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
