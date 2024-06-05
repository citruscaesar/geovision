from typing import Any, Optional 

import torch
from lightning import LightningModule
from geovision.config.basemodels import ExperimentConfig
from geovision.config.parsers import get_task

class ClassificationModule(LightningModule):
    def __init__(self, config: ExperimentConfig, model: Optional[torch.nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        self.learning_rate = config.optimizer_params.lr 
        self.batch_size = config.dataloader_params.batch_size // config.dataloader_params.gradient_accumulation
        if model is not None:
            self.model = model
        else:
            self.model = config.nn(
                num_classes = config.dataset.num_classes, 
                **config.nn_params.model_dump(exclude_none=True)
            )
        self.criterion = config.criterion(**config.criterion_params.model_dump(exclude_none=True))
        self.metric_params = {
            # TODO: add option for task to be 'multilabel' from config.dataset.name.split('_')[-1]
            "task": get_task(config.dataset),
            "num_classes": config.dataset.num_classes,
        }
        # TODO: add option for multiple metrics to be tracked, return MetricCollection from get_metric and treat config.metric as a list
        self.train_metric = config.get_metric(config.metric, self.metric_params)
        self.val_metric = config.get_metric(config.metric, self.metric_params)
        self.test_metric = config.get_metric(config.metric, self.metric_params)
    
    def configure_optimizers(self):
        optimizer_params = self.config.optimizer_params.model_dump(exclude_none=True)
        optimizer_params["params"] = self.model.parameters()
        optimizer = self.config.optimizer(**optimizer_params)
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
        return {"loss": loss, "preds": preds} 
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.log("val/loss", loss, on_step = True, on_epoch = True)
        self.log(f"val/{self.config.metric}", self.val_metric(preds, labels), on_step = True, on_epoch = True)
        return {"loss": loss, "preds": preds} 

    # def test_step(self, batch, batch_idx):
        # preds, labels, loss = self._forward(batch) 
        # self.log("test/loss", loss, on_step = True, on_epoch = True)
        # self.log(f"test/{self.config.metric}", self.test_metric(preds, labels), on_step = True, on_epoch = True)
        # return {"preds": preds, "loss": loss}