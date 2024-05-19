from pathlib import Path
from torchmetrics import ConfusionMatrix
import numpy as np
import pandas as pd

from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import Callback

from etl.etl import validate_dir
from training.evaluation import metrics_dict, plot_report 
from torchmetrics.functional import jaccard_index, f1_score

from typing import Any, Mapping, Optional, Literal
from torch import Tensor
from lightning import LightningModule, Trainer
from numpy.typing import NDArray
from matplotlib.figure import Figure
from pandas import DataFrame

class EvaluateClassification(Callback):
    def __init__(self) -> None:
        super().__init__()

    def __samples_df(self, samples: dict[str, list], num_classes: int) -> pd.DataFrame:
        class_names = [f"class_{c}" for c in range(num_classes)]
        df = pd.DataFrame(columns = ["idx", *class_names])
        for idx, pred in zip(samples["idx"], samples["preds"]):
            df.loc[len(df.index)] = [idx, *pred]
        df["idx"] = df["idx"].astype(np.uint32)
        return df

    def __on_eval_start(self, trainer: Trainer, dataset_df: DataFrame) -> None:
        self.csv_logger, self.wandb_logger = None, None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
                dataset_df.to_csv(Path(self.csv_logger.log_dir, "dataset.csv"), index = False)

            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)
    
    def __on_eval_epoch_start(self) -> None:
        self.samples = {"idx": list(), "preds": list()}
    
    def __on_eval_batch_end(self, outputs: Tensor, inputs: tuple[Tensor, Tensor, Tensor]) -> None:
        assert isinstance(inputs, tuple | list), f"expected type(input to eval_step) = list or tuple, got {type(inputs)}"
        assert isinstance(outputs, Tensor), f"expected type(output of eval_step) = Tensor, got {type(outputs)}"
        if len(inputs) > 2:
            self.samples["idx"].extend(inputs[2].tolist())
            self.samples["preds"].extend(outputs.detach().cpu().softmax(0).tolist())
    
    def __on_eval_epoch_end(self, trainer: Trainer, pl_module: LightningModule, confusion_matrix: ConfusionMatrix, prefix: Literal["val", "test"]) -> None:
        _step, _epoch = trainer.global_step, trainer.current_epoch
        _num_classes, _class_names = pl_module.num_classes, pl_module.class_names

        _samples: DataFrame = self.__samples_df(self.samples, _num_classes)
        _confm  : NDArray = confusion_matrix.compute().cpu().numpy() # type: ignore
        _fig    : Figure = plot_report(_confm, _class_names, _step, _epoch)
        _prefix : str = f"{prefix}/"

        pl_module.log_dict(metrics_dict(_confm, _prefix, _class_names))
        if self.csv_logger is not None:
            _eval_dir: Path = validate_dir(self.csv_logger.log_dir, "eval")
            _samples.to_csv(_eval_dir / f"epoch={_epoch}_step={_step}_{prefix}_samples.csv", index = False)
            _fig.savefig(_eval_dir / f"epoch={_epoch}_step={_step}_{prefix}_eval_report.png")
            np.save(_eval_dir / f"epoch={_epoch}_step={_step}_{prefix}_confusion_matrix.npy", _confm)
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = f"{_prefix}samples", dataframe = _samples, step = _step)
            self.wandb_logger.experiment.log({f"{_prefix}eval_report": _fig, "trainer/global_step": _step})
            self.wandb_logger.experiment.log({f"{_prefix}confusion_matrix": _confm, "trainer/global_step": _step})

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            self.__on_eval_start(trainer, trainer.datamodule.val_dataset.df) # type: ignore
        
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            self.__on_eval_epoch_start() 
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not trainer.sanity_checking:
            self.__on_eval_batch_end(outputs, batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            self.__on_eval_epoch_end(trainer, pl_module, pl_module.val_confusion_matrix, "val")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_start(trainer, trainer.datamodule.test_dataset.df) # type: ignore
        
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_start() 
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.__on_eval_batch_end(outputs, batch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_end(trainer, pl_module, pl_module.test_confusion_matrix, "test")

class EvaluateSegmentation(Callback):
    def __init__(self) -> None:
        super().__init__()

    def __on_eval_start(self, trainer: Trainer, dataset_df: DataFrame) -> None:
        self.csv_logger, self.wandb_logger = None, None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                self.csv_logger = logger
                dataset_df.to_csv(Path(self.csv_logger.log_dir, "dataset.csv"), index = False)

            elif isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.log_table(key = "dataset", dataframe = dataset_df, step = 0)

    def __on_eval_epoch_start(self) -> None:
        self.samples = {"idx": list(), "iou": list(), "dice": list()}
    
    def __on_eval_batch_end(self, outputs: Tensor, inputs: tuple[Tensor, Tensor, Tensor], num_classes: int) -> None:
        assert isinstance(inputs, tuple | list), f"expected type(input to eval_step) = list or tuple, got {type(inputs)}"
        assert isinstance(outputs, Tensor), f"expected type(output of eval_step) = Tensor, got {type(outputs)}"
        if len(inputs) > 2:
            preds, masks, idxs = outputs.argmax(1).detach().cpu(), inputs[1].argmax(1).detach().cpu(), inputs[2].detach().cpu()
            metric_kwargs = {"task" : "multiclass" if num_classes > 2 else "binary", "num_classes" : num_classes, "average": "macro"}
            for idx, pred, mask in zip(idxs, preds, masks):
                self.samples["idx"].append(idx.item())
                self.samples["iou"].append(jaccard_index(pred, mask, **metric_kwargs).item())
                self.samples["dice"].append(f1_score(pred, mask, **metric_kwargs).item())

    def __on_eval_epoch_end(self, trainer: Trainer, pl_module: LightningModule, confusion_matrix: ConfusionMatrix, prefix: Literal["val", "test"]) -> None:
        _step, _epoch = trainer.global_step, trainer.current_epoch
        _num_classes, _class_names = pl_module.num_classes, pl_module.class_names

        _samples: DataFrame = DataFrame(self.samples) 
        _confm  : NDArray = confusion_matrix.compute().cpu().numpy() # type: ignore
        _fig    : Figure = plot_report(_confm, _class_names, _step, _epoch)
        _prefix : str = f"{prefix}/"

        pl_module.log_dict(metrics_dict(_confm, _prefix, _class_names))
        if self.csv_logger is not None:
            _eval_dir: Path = validate_dir(self.csv_logger.log_dir, "eval")
            _samples.to_csv(_eval_dir / f"epoch={_epoch}_step={_step}_{prefix}_samples.csv", index = False)
            _fig.savefig(_eval_dir / f"epoch={_epoch}_step={_step}_{prefix}_eval_report.png")
            np.save(_eval_dir / f"epoch={_epoch}_step={_step}_{prefix}_confusion_matrix.npy", _confm)
        if self.wandb_logger is not None:
            self.wandb_logger.log_table(key = f"{_prefix}samples", dataframe = _samples, step = _step)
            self.wandb_logger.experiment.log({f"{_prefix}eval_report": _fig, "trainer/global_step": _step})
            self.wandb_logger.experiment.log({f"{_prefix}confusion_matrix": _confm, "trainer/global_step": _step})

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_start(trainer, trainer.datamodule.val_dataset.df) # type: ignore
        
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_start() 
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.__on_eval_batch_end(outputs, batch, pl_module.num_classes)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_end(trainer, pl_module, pl_module.val_confusion_matrix, "val")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_start(trainer, trainer.datamodule.test_dataset.df) # type: ignore
        
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_start() 
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.__on_eval_batch_end(outputs, batch, pl_module.num_classes)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__on_eval_epoch_end(trainer, pl_module, pl_module.test_confusion_matrix, "test")