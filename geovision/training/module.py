from typing import Any 

import torch
from lightning import LightningModule
from torchmetrics import MetricCollection
from geovision.config.basemodels import ExperimentConfig

class ClassificationModule(LightningModule):
    def __init__(self, config: ExperimentConfig, model: torch.nn.Module) -> None:
        super().__init__()
        
        self.config = config
        self.model = config.model(num_classes = config.dataset.num_classes, **config.model_params) #type: ignore
        self.criterion = config.criterion(**config.criterion_params) #type: ignore
        self._set_metrics()

    @property
    def class_names(self):
        return self.config.dataset.class_names 
    
    @property
    def num_classes(self):
        return self.config.dataset.num_classes

    @property
    def batch_size(self):
        return self.config.dataloader_params.batch_size // self.config.dataloader_params.gradient_accumulation
    
    @property
    def learning_rate(self):
        return self.config.optimizer_params["lr"]
    
    def configure_optimizers(self):
        self.config.optimizer_params["params"] = self.model.parameters()
        return self.config.optimizer(**self.config.optimizer_params)

    def _set_metrics(self) -> None:
        monitor = self.config.metric
        metric_params = {
            "task" : "multiclass" if self.num_classes > 2 else "binary",
            "num_classes": self.num_classes,
        }

        metrics = MetricCollection({
            monitor: self.config.get_metric(monitor, metric_params),
            # NOTE: "cohen_kappa": get_metric("cohen_kappa", metric_params),
        })

        self.train_metrics = metrics.clone(prefix = "train/")
        self.val_metrics = metrics.clone(prefix = "val/")
        self.test_metrics = metrics.clone(prefix = "test/")

        self.val_losses = list()
        self.test_losses = list()
        self.val_confusion_matrix = self.config.get_metric("confusion_matrix", metric_params)
        self.test_confusion_matrix = self.config.get_metric("confusion_matrix", metric_params)       
    
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



