from typing import Any, Mapping, Literal
from numpy.typing import NDArray

import h5py
import torch
import torchmetrics
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import count
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from geovision.data.dataset import Dataset
from geovision.logging import get_logger
from geovision.io.local import get_new_dir, get_experiments_dir
from .metrics import get_classification_metrics_df, get_classification_metrics_dict

class ModelOutputLogger:
    def __init__(self, split: Literal["val", "test"], config, trainer: Trainer, pl_module: LightningModule):
        match split:
            case "val":
                self.dataset = trainer.val_dataloaders.dataset
            case "test":
                self.dataset = trainer.test_dataloaders.dataset
            case _:
                raise ValueError(f":split must be one of val or test, got {split}")
        self.split = split
        self.config = config
        self.experiments_dir = get_experiments_dir(config)
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

    def log_confusion_matrix(self):
        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds = self.get_probs(),
            target = self.get_labels(),
            task = "multiclass" if self.config.dataset.num_classes > 2 else "binary",
            num_classes = self.config.dataset.num_classes,
        ).numpy().astype(np.uint8)
        with h5py.File(self.output_file_path, "r+") as f:
            f.create_dataset("confm", data = confusion_matrix)

    def log_metrics_df(self):
        metrics_df = get_classification_metrics_df(self.get_confusion_matrix(), self.config.dataset.class_names)
        metrics_df.to_hdf(self.output_file_path, key = "metrics_df", mode = "r+")
    
    def log_metrics(self):
        self.log_confusion_matrix()
        self.log_metrics_df()

    def get_probs(self) -> torch.Tensor:
        with h5py.File(self.output_file_path, "r") as f:
            return torch.tensor(f["probs"][:])
    
    def get_labels(self) -> torch.Tensor:
        with h5py.File(self.output_file_path, "r") as f:
            return torch.tensor(f["labels"][:])

    def get_df_idxs(self) -> torch.Tensor:
        with h5py.File(self.output_file_path, "r") as f:
            return torch.tensor(f["df_idxs"][:])
    
    def get_confusion_matrix(self) -> NDArray:
        with h5py.File(self.output_file_path, "r") as f:
            return f["confm"][:]
    
    def get_metrics_df(self) -> pd.DataFrame:
        return pd.read_hdf(self.output_file_path, "metrics_df", "r")
    
    def get_metrics_dict(self) -> dict:
        metrics_dict = get_classification_metrics_dict(self.get_metrics_df())
        return {f"{self.split}/{k}":v for k,v in metrics_dict.items()}

class ClassificationMetricsLogger(Callback):
    # store model output logits in memory in a pre-reserved numpy array during eval step
    # save model outputs and compute confusion matrix and other matrices during eval epoch end 
    # log everything to CSVLogger and WandbLogger during eval end

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experiments_dir = get_experiments_dir(config)
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.csv_logger, self.wandb_logger = None, None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger

        dataset_df: pd.DataFrame
        if hasattr(trainer.datamodule, "train_dataset"):
            dataset_df = trainer.datamodule.train_dataset.df
        elif hasattr(trainer.datamodule, "val_dataset"):
            dataset_df = trainer.datamodule.val_dataset.df
        elif hasattr(trainer.datamodule, "test_dataset"):
            dataset_df = trainer.datamodule.test_dataset.df
        else:
            raise Exception("couldn't load df from datamodule.train/val/test_dataset") 

        if self.csv_logger is not None:        
            dataset_df.to_csv(self.experiments_dir/"dataset.csv", index = False)
        if self.wandb_logger is not None:
            self.wandb_logger.log_table("dataset", dataframe = dataset_df)
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_logger = ModelOutputLogger("val", self.config, trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_logger = ModelOutputLogger("test", self.config, trainer, pl_module)
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.output_logger.log(outputs["preds"], batch)
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.output_logger.log(outputs["preds"], batch)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_logger.log_metrics()
        pl_module.log_dict(self.output_logger.get_metrics_dict())

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_logger.log_metrics()
        pl_module.log_dict(self.output_logger.get_metrics_dict())

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.wandb_logger is not None:
            import wandb
            wandb.log({f"val/confm_epoch={trainer.current_epoch}_step={trainer.global_step}" : wandb.plot.confusion_matrix(
                probs = self.output_logger.get_probs().numpy(),
                y_true = self.output_logger.get_labels().numpy(),
                class_names = self.config.dataset.class_names)
            })

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.wandb_logger is not None:
            import wandb
            wandb.log({f"test/confm_epoch={trainer.current_epoch}_step={trainer.global_step}" : wandb.plot.confusion_matrix(
                probs = self.output_logger.get_probs().numpy(),
                y_true = self.output_logger.get_labels().numpy(),
                class_names = self.config.dataset.class_names)
            })

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