from typing import Any, Mapping, Literal

import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from geovision.data.dataset import Dataset
from geovision.logging import get_logger
from geovision.io.local import get_new_dir, get_experiments_dir

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

        #pre-allocate nxk array

    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logits_file_path = self.get_logits_file_path("val", trainer)
        self.create_logits_file("val", trainer)
        self.logs_dir = self.logits_file_path.parent

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logits_file_path = self.get_logits_file_path("test", trainer)
        self.create_logits_file("test", trainer)
        self.logs_dir = self.logits_file_path.parent
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.write_batch_to_logits_file(outputs, batch, batch_idx)
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.write_batch_to_logits_file(outputs, batch, batch_idx)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logits, labels, df_idxs = self.read_from_logits_file()
        confusion_matrix = torchmetrics.functional.confusion_matrix(logits, labels, )

    def get_logits_file_path(self, mode: Literal["val", "test"], trainer: Trainer):
        return get_new_dir(self.experiments_dir, f"{mode}_logs") / f"epoch={trainer.current_epoch}_step={trainer.global_step}_logits.h5"

    def create_logits_file(self, mode: Literal["val", "test"], trainer: Trainer):
        match mode:
            case "val":
                dataset = trainer.val_dataloaders.dataset # type: ignore
            case "test":
                dataset = trainer.test_dataloaders.dataset # type: ignore
            case _:
                raise ValueError("mode must be either val or test")

        self.logits_file_idx = 0
        num_eval_samples = len(dataset)
        num_output_classes = dataset.num_classes
        with h5py.File(self.logits_file_path, 'w') as f:
            f.create_dataset("logits", (num_eval_samples, num_output_classes), np.float32)
            f.create_dataset("labels", (num_eval_samples), np.uint8)
            f.create_dataset("df_idxs", (num_eval_samples), np.uint8)
    
    def write_batch_to_logits_file(self, outputs, batch, batch_idx):
        assert isinstance(outputs, torch.Tensor), f"expected a Tensor, got {type(outputs)}"
        logits = torch.softmax(outputs, 1).detach().cpu().numpy().astype(np.float32)
        labels = batch[1].cpu().numpy().astype(np.uint8)
        df_idxs = batch[2].cpu().numpy().astype(np.uint8)
        print(f"batch #{batch_idx}")
        with h5py.File(self.logits_file_path, "r+") as f:
            for i in range(len(logits)):
                f["logits"][self.logits_file_idx] = logits[i]
                f["labels"][self.logits_file_idx] = labels[i]
                f["df_idxs"][self.logits_file_idx] = df_idxs[i]
                self.logits_file_idx += 1
    
    def read_from_logits_file(self):
        with h5py.File(self.logits_file_path, "r") as f:
            return f["logits"][:], f["labels"][:], f["df_idxs"][:]
    
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