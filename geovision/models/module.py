from typing import Any

import torch
from lightning import LightningModule
from geovision.config.config import ExperimentConfig

class ClassificationModule(LightningModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.model = config.get_nn()
        self.criterion = config.get_criterion()
        self.metric = config.get_metric()
    
    def configure_optimizers(self):
        optimizer = self.config.get_optimizer(self.model.parameters())
        scheduler = self.config.get_scheduler(optimizer)
        if scheduler is not None:
            return {"optimizer": optimizer, "scheduler": scheduler} 
        return optimizer
    
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
        self.log("train/loss", loss, on_step = False, on_epoch = True)
        self.log(f"train/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        self.metric.reset()
        return {"loss": loss, "preds": preds} 
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.log("val/loss", loss, on_step = False, on_epoch = True)
        self.log(f"val/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        self.metric.reset()
        return {"loss": loss, "preds": preds} 

    def test_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch) 
        self.test_metric.update(preds, labels)
        self.log("test/loss", loss, on_step = False, on_epoch = True)
        self.log(f"test/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        self.metric.reset()
        return {"loss": loss, "preds": preds} 