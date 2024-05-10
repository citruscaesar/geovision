from typing import Any, Optional 

import torch
from lightning import LightningModule
from torchmetrics import MetricCollection
from geovision.config import ExperimentConfig
from geovision.config.registers import (
    get_dataset, 
    get_model, 
    get_criterion,
    get_metric,
    get_optimizer
)

class ClassificationModule(LightningModule):
    def __init__(self, config: ExperimentConfig, model: Optional[torch.nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        if model is None: 
            self.model = get_model(config.model.name, (config.model.params | {"num_classes": self.num_classes})) 
        else:
            self.model = model
        self.criterion = get_criterion(config.loss.name, config.loss.params) 
        #self._set_metrics()

    @property
    def class_names(self):
        return get_dataset(self.config.dataset.name).class_names
    
    @property
    def num_classes(self):
        return get_dataset(self.config.dataset.name).num_classes

    @property
    def batch_size(self):
        return self.config.dataloader.batch_size // self.config.dataloader.gradient_accumulation
    
    @property
    def learning_rate(self):
        return self.config.optimizer.params["lr"]
    
    def configure_optimizers(self):
        self.config.optimizer.params["params"] = self.model.parameters()
        return get_optimizer(self.config.optimizer.name, **self.config.optimizer.params)

    def _set_metrics(self) -> None:
        # print(f"monitor metric: {monitor_metric}")
        monitor = self.config.metric.name
        metric_params = {
            "task" : "multiclass" if self.num_classes > 2 else "binary",
            "num_classes": self.num_classes,
        }

        metrics = MetricCollection({
            monitor: get_metric(monitor, **metric_params),
            # NOTE: "cohen_kappa": get_metric("cohen_kappa", metric_params),
        })

        self.train_metrics = metrics.clone(prefix = "train/")
        self.val_metrics = metrics.clone(prefix = "val/")
        self.test_metrics = metrics.clone(prefix = "test/")

        self.val_losses = list()
        self.test_losses = list()
        self.val_confusion_matrix = get_metric("confusion_matrix", **metric_params)
        self.test_confusion_matrix = get_metric("confusion_matrix", **metric_params)       
    
    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: batch_size: N, num_channels: C, height: H, width: W, num_classes: C' 
        # NOTE: Classification: (NCHW) -> model -> (NC') -> argmax(1) -> (N,)
        # NOTE: Segmentation: (NCHW) -> model -> (NC'HW) -> argmax(1) -> (NHW)
        images, labels = batch[0], batch[1]
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        return preds, labels, loss

    def forward(self, batch) -> Any:
        images = batch[0]
        images.requires_grad = True
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.train_metrics.update(preds, labels)

        # NOTE: "train/loss" must have on_step = True and on_epoch = True
        self.log("train/loss", loss, on_epoch = True, on_step = True, batch_size = self.batch_size)
        self.log("lr", self.learning_rate, on_step = False, on_epoch = True)
        self.log_dict(self.train_metrics, on_epoch = True, batch_size = self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.val_losses.append(loss)
        self.val_metrics.update(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        return preds 

    def on_validation_epoch_end(self):
        self.log("val/loss", torch.tensor(self.val_losses).mean())
        self.log_dict(self.val_metrics.compute())
        self.val_losses.clear()
        self.val_metrics.reset()
        self.val_confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.test_losses.append(loss)
        self.test_metrics.update(preds, labels)
        self.test_confusion_matrix.update(preds, labels)
        return preds 

    def on_test_epoch_end(self):
        self.log("test/loss", torch.tensor(self.test_losses).mean())
        self.log_dict(self.test_metrics.compute())
        self.test_losses.clear()
        self.test_metrics.reset()
        self.test_confusion_matrix.reset()



