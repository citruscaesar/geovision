from typing import Any 

import torch
from lightning import LightningModule
from torchmetrics import MetricCollection
from geovision.config.basemodels import ExperimentConfig

from torchvision.models import alexnet, AlexNet_Weights

class ClassificationModule(LightningModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        
        self.config = config
        self.model = config.nn(num_classes = config.dataset.num_classes, **config.nn_params.model_dump()) #type: ignore
        #self.model = alexnet(weights = AlexNet_Weights.IMAGENET1K_V1) 
        #self.model.classifier[-1] = torch.nn.Linear(4096, 10)
        self.criterion = config.criterion(**config.criterion_params) #type: ignore

        self.metric_params = {
            # TODO: add option for task to be 'multilabel' from config.dataset.name.split('_')[-1]
            "task": "multiclass" if self.config.dataset.num_classes > 2 else "binary",
            "num_classes": self.config.dataset.num_classes,
        }
        # TODO: add option for multiple metrics to be tracked, return MetricCollection from get_metric and treat config.metric as a list
        self.train_metric = config.get_metric(config.metric, self.metric_params)
        self.val_metric = config.get_metric(config.metric, self.metric_params)
        self.test_metric = config.get_metric(config.metric, self.metric_params)

    @property
    def batch_size(self):
        return self.config.dataloader_params.batch_size // self.config.dataloader_params.gradient_accumulation
    
    @property
    def learning_rate(self):
        return self.config.optimizer_params["lr"]
    
    def configure_optimizers(self):
        self.config.optimizer_params["params"] = self.model.parameters()
        optimizer = self.config.optimizer(**self.config.optimizer_params)
        # lr_scheduler = {"scheduler": None, "monitor": self.config.metric, "frequency": None} 
        return optimizer #, [lr_scheduler]
    
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
        self.train_metric.update(preds, labels)
        self.log("train/loss", loss, on_step = True, on_epoch = True)
        self.log(f"train/{self.config.metric}", self.train_metric, on_step = True, on_epoch = True)
        return {"preds": preds, "loss": loss}
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.log("val/loss", loss, on_step = True, on_epoch = True)
        self.log(f"val/{self.config.metric}", self.val_metric(preds, labels), on_step = True, on_epoch = True)
        return {"preds": preds, "loss": loss}

    # def test_step(self, batch, batch_idx):
        # preds, labels, loss = self._forward(batch) 
        # self.log("test/loss", loss, on_step = True, on_epoch = True)
        # self.log(f"test/{self.config.metric}", self.test_metric(preds, labels), on_step = True, on_epoch = True)
        # return {"preds": preds, "loss": loss}

    # self.log("lr", self.learning_rate, on_step = False, on_epoch = True)
    # self.log_dict(self.train_metrics, on_epoch = True, batch_size = self.batch_size)
    # def validation_step(self, batch, batch_idx):
        # preds, labels, loss = self._forward(batch) 
        # self.val_losses.append(loss)
        # self.val_metrics.update(preds, labels)
        # self.val_confusion_matrix.update(preds, labels)
        # return preds 

    # def on_validation_epoch_end(self):
        # self.log("val/loss", torch.tensor(self.val_losses).mean())
        # self.log_dict(self.val_metrics.compute())
        # self.val_losses.clear()
        # self.val_metrics.reset()
        # self.val_confusion_matrix.reset()

    # def test_step(self, batch, batch_idx):
        # preds, labels, loss = self._forward(batch) 
        # self.test_losses.append(loss)
        # self.test_metrics.update(preds, labels)
        # self.test_confusion_matrix.update(preds, labels)
        # return preds 

    # def on_test_epoch_end(self):
        # self.log("test/loss", torch.tensor(self.test_losses).mean())
        # self.log_dict(self.test_metrics.compute())
        # self.test_losses.clear()
        # self.test_metrics.reset()
        # self.test_confusion_matrix.reset()



