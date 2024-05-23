from typing import Any, Mapping, Literal

import h5py
import torch
import torchmetrics
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import count
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from geovision.data.dataset import Dataset
from geovision.logging import get_logger
from geovision.io.local import get_new_dir, get_experiments_dir
from .metrics import get_classification_metrics_df

class ModelOutputLogger:
    def __init__(self, split: Literal["val", "test"], trainer: Trainer, pl_module: LightningModule):
        match split:
            case "val":
                self.dataset = trainer.val_dataloaders.dataset
            case "test":
                self.dataset = trainer.test_dataloaders.dataset
            case _:
                raise ValueError(f":split must be one of val or test, got {split}")
        self.config = pl_module.config
        self.experiments_dir = get_experiments_dir(pl_module.config)
        self.output_file_path = get_new_dir(self.experiments_dir, f"{split}_logs") / f"epoch={trainer.current_epoch}_step={trainer.global_step}_outputs.h5"
        self.output_file_idx = count()
        self.setup_logfile()

    def setup_logfile(self):
        num_eval_samples = len(self.dataset)
        num_output_classes = self.dataset.num_classes
        with h5py.File(self.output_file_path, 'w') as f:
            f.create_dataset("probs", (num_eval_samples, num_output_classes), np.float32)
            f.create_dataset("labels", (num_eval_samples), np.uint8)
            f.create_dataset("df_idxs", (num_eval_samples), np.uint8)
    
    def log(self, outputs: torch.Tensor, batch: tuple):
        probs = torch.softmax(outputs, 1).detach().cpu().numpy().astype(np.float32)
        labels = batch[1].cpu().numpy().astype(np.uint8)
        df_idxs = batch[2].cpu().numpy().astype(np.uint8)
        with h5py.File(self.output_file_path, "r+") as f:
            for i, out_idx in zip(range(len(probs)), self.output_file_idx):
                f["probs"][out_idx] = probs[i]
                f["labels"][out_idx] = labels[i]
                f["df_idxs"][out_idx] = df_idxs[i]

    @property
    def probs(self) -> torch.Tensor:
        with h5py.File(self.output_file_path, "r") as f:
            return torch.tensor(f["probs"][:])
    
    @property
    def labels(self) -> torch.Tensor:
        with h5py.File(self.output_file_path, "r") as f:
            return torch.tensor(f["labels"][:])

    @property
    def df_idxs(self) -> torch.Tensor:
        with h5py.File(self.output_file_path, "r") as f:
            return torch.tensor(f["df_idxs"][:])

class ClassificationMetricsLogger(Callback):
    # store model output logits in memory in a pre-reserved numpy array during eval step
    # save model outputs and compute confusion matrix and other matrices during eval epoch end 
    # log everything to CSVLogger and WandbLogger during eval end

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiments_dir = get_experiments_dir(config)
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # setup loggers
        self.csv_logger, self.wandb_logger = None, None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_logger = ModelOutputLogger("val", trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_logger = ModelOutputLogger("test", trainer, pl_module)
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.output_logger.log(outputs["preds"], batch)
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.output_logger.log(outputs["preds"], batch)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds = self.output_logger.probs,
            target = self.output_logger.labels,
            task = "multiclass" if self.config.dataset.num_classes > 2 else "binary",
            num_classes = self.config.dataset.num_classes,
        )
        metrics_df = get_classification_metrics_df(confusion_matrix.numpy(), self.config.dataset.class_names)
        print(trainer.current_epoch, trainer.global_step)
        display(metrics_df)

class SegmentationMetricsLogger(Callback):
    pass

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
        monitor = f"val/{config.metric}",
        mode = "min" if (config.metric in min_metrics) else "max",
        save_top_k = save_top_k,
        every_n_epochs = None,
        every_n_train_steps = None,
        train_time_interval = None,
        save_on_train_epoch_end = False,
        enable_version_counter = True
    )

def get_classification_logger(config):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging classification metrics to {experiments_dir}")
    return ClassificationMetricsLogger(config)

def get_segmentation_logger(config):
    experiments_dir = get_experiments_dir(config)
    info_logger = get_logger("metrics_logger")
    info_logger.info(f"logging segmentation metrics to {experiments_dir}")
    return SegmentationMetricsLogger(config)