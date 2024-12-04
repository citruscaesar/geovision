from typing import Any, Optional  
from collections.abc import Callable

import torch
import lightning

from itertools import chain
from geovision.io.local import FileSystemIO as fs

# TODO: add option to selectively freeze model layers
# TODO: add option to provide different lr for different parts of the model

class ModelConfig:
    def __init__(
            self,
            encoder_name: str, 
            encoder_params: dict[str, Any],
            decoder_name: str,
            decoder_params: dict[str, Any],
            ckpt_path: Optional[str] = None
        ):
        self.encoder_constructor = self._get_encoder_constructor(encoder_name)
        self.decoder_constructor = self._get_decoder_constructor(decoder_name)

        # get ckpt path
        if ckpt_path is not None:
            self.ckpt_path = fs.get_valid_file_err(ckpt_path)
            # ckpt: dict = torch.load(self.ckpt_path)
            # assert ckpt["encoder_name"] == encoder_name, f"config error (name mismatch), expected encoder named {ckpt["encoder_name"]}"
            # assert ckpt["decoder_name"] == decoder_name, f"config error (name mismatch), expected decoder named {ckpt["decoder_name"]}"
            # del ckpt
        else:
            self.ckpt_path = None

        assert isinstance(encoder_params, dict), f"config error (invalid type), expected :encoder_params to be dict, got {type(encoder_params)}"
        #if self.ckpt_path is None:
            #assert encoder_params.get("weights") is not None, \
                #"config error, :weights cannot be None"
            #weights_path = encoder_params.get("weights_path")
            #if weights_init is not None:
                #assert weights_path is None, "config error, expected either :weights_init or :weights_path to be provided, got both"
                #assert isinstance(weights_init, str), f"config error (invalid type), expected :weights_init to be str, got {type(weights_init)}"
                #assert weights_init in self.encoder_constructor.weight_init_strategies, "config error (not impl.)"
            #else: 
                #assert weights_path is not None, "config error, expected either :weights_init or :weights_path to be provided, got neither"
                #assert fs.is_valid_file(weights_path), f"config error (invalid path), {weights_path} doesn't point to a valid location on local fs "
        self.encoder_name = encoder_name 
        self.encoder_params = encoder_params  

        assert isinstance(decoder_params, dict), f"config error (invalid type), expected :decoder_params to be dict, got {type(decoder_params)}"
        self.decoder_name = decoder_name 
        self.decoder_params = decoder_params
    
        # TODO: check decoder params

    def _get_encoder_constructor(self, name: str) -> torch.nn.Module:
        match name:
            case "resnet":
                from .resnet import ResNetFeatureExtractor
                return ResNetFeatureExtractor 
            case _:
                raise AssertionError(f"config error (not implemented), got {name}")

    def _get_decoder_constructor(self, name: str) -> torch.nn.Module:
        match name:
            case "linear":
                return torch.nn.Linear
            case "lazy_linear":
                return torch.nn.LazyLinear
            case _:
                raise AssertionError(f"config error (not implemented), got {name}")

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

        self.save_hyperparameters({
            "encoder": model_config.encoder_name, 
            "encoder_params": model_config.encoder_params,
            "decoder": model_config.decoder_name,
            "decoder_params": model_config.decoder_params,
            "criterion": criterion_constructor.__name__,
            "criterion_params": criterion_params,
            "optimizer": optimizer_constructor.__name__,
            "optimizer_params": optimizer_params,
            "lr_scheduler": lr_scheduler_constructor.__name__ if lr_scheduler_constructor is not None else '',
            "lr_scheduler_params": lr_scheduler_params,
            "warmup_scheduler": warmup_scheduler_constructor.__name__ if warmup_scheduler_constructor is not None else '',
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

    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: batch_size: N, num_channels: C, height: H, width: W, num_classes: C'
        # NOTE: Classification: (NCHW) -> model -> (NC') -> argmax(1) -> (N,)
        # NOTE: Segmentation: (NCHW) -> model -> (NC'HW) -> argmax(1) -> (NHW)
        images, labels = batch[0], batch[1]
        preds = self.forward(images) 
        loss = self.criterion(preds, labels)
        return preds, labels, loss

    def forward(self, images: torch.Tensor) -> Any:
        return self.decoder(self.encoder(images).flatten(1))

    def training_step(self, batch, batch_idx):
        preds, _, loss = self._forward(batch)
        # print(*self.lr_schedulers().get_last_lr())
        # self.log("train/loss", loss, on_step = False, on_epoch = True)
        # self.log(f"train/{self.config.metric}", self.metric(preds, labels), on_step = False, on_epoch = True, prog_bar=True)
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
