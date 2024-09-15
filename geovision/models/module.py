from typing import Any

import torch
from lightning import LightningModule
from geovision.config.experiment_config import ExperimentConfig

class ClassificationModule(LightningModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config

        self.model = self.config.model_constructor(**self.config.model_params)
        self.criterion = self.config.criterion_constructor(**self.config.criterion_params)
        # self.metric = config.metric_constructor(**self.metric_params)

    def configure_optimizers(self) -> dict[str, torch.optim.optimizer.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
        optimizer = self.config.optimizer_constructor(params = self.model.parameters(), **self.config.optimizer_params)
        if self.config.scheduler_name is None:
            return {"optimizer": optimizer}
        scheduler = self.config.scheduler_constructor(optimizer = optimizer, **self.config.scheduler_params)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler" : scheduler} | self.config.scheduler_config_params} 
    
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
        # self.log("train/loss", loss, on_step = False, on_epoch = True)
        # self.log(f"train/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        # self.metric.reset()
        return {"loss": loss, "preds": preds} 
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        # self.log("val/loss", loss, on_step = False, on_epoch = True)
        # self.log(f"val/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        # self.metric.reset()
        return {"loss": loss, "preds": preds} 

    def test_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        # self.test_metric.update(preds, labels)
        # self.log("test/loss", loss, on_step = False, on_epoch = True)
        # self.log(f"test/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        # self.metric.reset()
        return {"loss": loss, "preds": preds} 