from typing import Any, Optional, Literal
from collections.abc import Callable

import torch
import lightning
import torchmetrics

from itertools import chain
from importlib import import_module
from geovision.models import blocks
from geovision.io.local import FileSystemIO as fs

# TODO: add option to selectively freeze model layers
# TODO: add option to provide different lr for different parts of the model

class ModelConfig:
    def __init__(
            self,
            encoder: str, 
            encoder_params: dict[str, Any],
            decoder: str,
            decoder_params: dict[str, Any],
            ckpt_path: Optional[str] = None
        ):
        self.encoder_constructor = self._get_constructor(encoder)
        self.decoder_constructor = self._get_constructor(decoder)

        assert isinstance(encoder_params, dict), f"config error (invalid type), expected :encoder_params to be dict, got {type(encoder_params)}"
        for k in encoder_params.keys():
            if k.endswith("_block"):
                encoder_params[k] = getattr(blocks, encoder_params[k])
            if k == "weights":
                encoder_params[k] = self._get_weights_enum(encoder_params[k])
        self.encoder_params = encoder_params 

        assert isinstance(decoder_params, dict), f"config error (invalid type), expected :decoder_params to be dict, got {type(decoder_params)}"
        for k in decoder_params.keys():
            if k.endswith("_block"):
                decoder_params[k] = getattr(blocks, decoder_params[k])
        self.decoder_params = decoder_params

        if ckpt_path is not None:
            self.ckpt_path = fs.get_valid_file_err(ckpt_path)
        else:
            self.ckpt_path = None
    
    def _get_weights_enum(self, name: str):
        name: list = name.split('.')
        get_weight: callable = getattr(import_module('.'.join(name[:2])), "get_weight")
        return get_weight('.'.join(name[-2:]))

    def _get_constructor(self, name: str) -> torch.nn.Module:
        name = name.split('.') 
        return getattr(import_module('.'.join(name[:-1])), name[-1])
    
class ClassificationModule(lightning.LightningModule):
    def __init__(
            self,
            model_config: ModelConfig,
            criterion_constructor: Callable[..., torch.nn.Module],
            criterion_params: dict,
            optimizer_constructor: Callable[..., torch.optim.Optimizer],
            optimizer_params: dict,
            lr_scheduler_constructor: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
            lr_scheduler_params: Optional[dict] = None,
            warmup_scheduler_constructor: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
            warmup_steps: Optional[int] = None,
            warmup_scheduler_params: Optional[dict] = None,
            scheduler_config_params: Optional[dict] = None,
    ) -> None:

        super().__init__()
        self.encoder = model_config.encoder_constructor(**model_config.encoder_params)

        if "Linear" in model_config.decoder_constructor.__qualname__:
            self.forward = self._clf_forward

        elif "UNet" in model_config.decoder_constructor.__qualname__:
            model_config.decoder_params["layer_ch"] = self.encoder._out_ch_per_layer 
            model_config.decoder_params["layer_up"] = self.encoder._downsampling_per_layer
            self.forward = self._unet_forward

        else:
            NotImplementedError(f"expected :decoder to be Linear or Unet, got {model_config.decoder_constructor.__qualname__}")
        
        self.decoder = model_config.decoder_constructor(**model_config.decoder_params)
        self.criterion = criterion_constructor(**criterion_params)

        self.optimizer_constructor = optimizer_constructor
        self.optimizer_params = optimizer_params
        self.lr_scheduler_constructor = lr_scheduler_constructor
        self.lr_scheduler_params = lr_scheduler_params
        self.warmup_scheduler_constructor = warmup_scheduler_constructor
        self.warmup_steps = warmup_steps
        self.warmup_scheduler_params = warmup_scheduler_params
        self.scheduler_config_params = scheduler_config_params

        if model_config.ckpt_path is not None:
            self.load_from_checkpoint(model_config.ckpt_path)
        
        self.train_acc = torchmetrics.Accuracy("binary", num_classes = 2, sync_on_compute = True)

        self.save_hyperparameters({
            "encoder": model_config.encoder_constructor.__qualname__, 
            "encoder_params": model_config.encoder_params,
            "decoder": model_config.decoder_constructor.__qualname__,
            "decoder_params": model_config.decoder_params,
            "criterion": criterion_constructor.__qualname__,
            "criterion_params": criterion_params,
            "optimizer": optimizer_constructor.__qualname__,
            "optimizer_params": optimizer_params,
            "lr_scheduler": lr_scheduler_constructor.__qualname__ if lr_scheduler_constructor is not None else '',
            "lr_scheduler_params": lr_scheduler_params,
            "warmup_scheduler": warmup_scheduler_constructor.__qualname__ if warmup_scheduler_constructor is not None else '',
            "warmup_scheduler_params": warmup_scheduler_params,
            "warmup_steps": warmup_steps,
            "scheduler_config_params": scheduler_config_params
        })

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
        def get_lr_scheduler():
            return self.lr_scheduler_constructor(config["optimizer"], **self.lr_scheduler_params)

        def get_warmup_scheduler():
            return self.warmup_scheduler_constructor(config["optimizer"], **self.warmup_scheduler_params)

        def get_warmup_and_lr_scheduler():
            return torch.optim.lr_scheduler.SequentialLR(config["optimizer"], [get_warmup_scheduler(), get_lr_scheduler()], [self.warmup_steps])

        config = {"optimizer": self.optimizer_constructor(chain(self.encoder.parameters(), self.decoder.parameters()), **self.optimizer_params)}

        if self.lr_scheduler_constructor is not None:
            if self.warmup_scheduler_constructor is not None:
                config["lr_scheduler"] = {"scheduler": get_warmup_and_lr_scheduler()} | self.scheduler_config_params 
            else:
                config["lr_scheduler"] = {"scheduler": get_lr_scheduler()} | self.scheduler_config_params 
        elif self.warmup_scheduler_constructor is not None:
            config["lr_scheduler"] = {"scheduler": get_warmup_scheduler()} | self.scheduler_config_params 

        return config
    
    def _clf_forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))
    
    def _unet_forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder_outputs = list() 
        for layer in self.encoder.children():
            images = layer(images)
            encoder_outputs.append(images)
        return self.decoder(*reversed(encoder_outputs))

    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: batch_size: N, num_channels: C, height: H, width: W, num_classes: C'
        # NOTE: Classification: (NCHW) -> model -> (NC') -> argmax(1) -> (N,)
        # NOTE: Segmentation: (NCHW) -> model -> (NC'HW) -> argmax(1) -> (NHW)
        images, labels = batch[0], batch[1]
        preds = self.forward(images) 
        loss = self.criterion(preds, labels)
        return preds, labels, loss

    def training_step(self, batch, batch_idx):
        preds, _, loss = self._forward(batch)
        # print(*self.lr_schedulers().get_last_lr())
        #self.log("train/loss", loss, on_step = False, on_epoch = True)
        #self.log("train/accuracy", self.train_acc(preds, labels), on_step = False, on_epoch = True, prog_bar = True)
        # self.metric.reset()
        return {"loss": loss, "preds": preds}

    def validation_step(self, batch, batch_idx):
        preds, _, loss = self._forward(batch)
        # self.log("val/loss", loss, on_step = False, on_epoch = True)
        # self.log(f"val/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        # self.metric.reset()
        return {"loss": loss, "preds": preds}

    def test_step(self, batch, batch_idx):
        preds, _, loss = self._forward(batch)
        # self.test_metric.update(preds, labels)
        # self.log("test/loss", loss, on_step = False, on_epoch = True)
        # self.log(f"test/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
        # self.metric.reset()
        return {"loss": loss, "preds": preds}