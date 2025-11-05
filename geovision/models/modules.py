from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Callable

import torch
import lightning
import importlib

from itertools import chain
from geovision.io.local import FileSystemIO as fs
from geovision.experiment.config import Builder, ExperimentConfig

# TODO: add option to selectively freeze model layers
# TODO: add option to provide different lr for different parts of the model
    
class ClassificationModule(lightning.LightningModule):
    def __init__(self, config: ExperimentConfig): 
        super().__init__()
        self.config = config

        self.encoder = Builder(**config.module["kwargs"]["model_encoder"])
        self.decoder = Builder(**config.module["kwargs"]["model_decoder"])

        self.decoder.kwargs["in_ch"] = self.encoder.constructor._out_ch_per_layer[-1]
        self.decoder.kwargs["out_ch"] = config.dataset.constructor.num_classes

        self.encoder = self.encoder.build()
        self.decoder = self.decoder.build()
        self.criterion = config.criterion.build() # nn.Module

        # # if decoder_kwargs is not None:
            # # decoder_kwargs["out_ch"] = self.dataset_constructor.num_classes

        #if "Linear" in self.model_config.decoder_constructor.__qualname__:
            #self.model_config.decoder_params["in_ch"] = self.encoder._out_ch_per_layer[-1]
            #self.forward = self._clf_forward

        #else:
            #NotImplementedError(f"expected :decoder to be Linear or Unet, got {model_config.decoder_constructor.__qualname__}")
        
        #self.decoder = model_config.decoder_constructor(**model_config.decoder_params)
        #self.criterion = model_config.criterion_constructor(**criterion_params)

        # self.optimizer_constructor = optimizer_constructor
        # self.optimizer_params = optimizer_params
        # self.lr_scheduler_constructor = lr_scheduler_constructor
        # self.lr_scheduler_params = lr_scheduler_params
        # self.warmup_scheduler_constructor = warmup_scheduler_constructor
        # self.warmup_steps = warmup_steps
        # self.warmup_scheduler_params = warmup_scheduler_params
        # self.scheduler_config_params = scheduler_config_params

        # NOTE: does this even work? 
        # if model_config.ckpt_path is not None:
            # self.load_from_checkpoint(model_config.ckpt_path)
        
        # NOTE: no real need, since we'll copy the config to the experiments dir anyway
        # self.save_hyperparameters({
            # "encoder": model_config.encoder_constructor.__qualname__, 
            # "encoder_params": model_config.encoder_params,
            # "decoder": model_config.decoder_constructor.__qualname__,
            # "decoder_params": model_config.decoder_params,
            # "criterion": criterion_constructor.__qualname__,
            # "criterion_params": criterion_params,
            # "optimizer": optimizer_constructor.__qualname__,
            # "optimizer_params": optimizer_params,
            # "lr_scheduler": lr_scheduler_constructor.__qualname__ if lr_scheduler_constructor is not None else '',
            # "lr_scheduler_params": lr_scheduler_params,
            # "warmup_scheduler": warmup_scheduler_constructor.__qualname__ if warmup_scheduler_constructor is not None else '',
            # "warmup_scheduler_params": warmup_scheduler_params,
            # "warmup_steps": warmup_steps,
            # "scheduler_config_params": scheduler_config_params
        # })

    def configure_optimizers(self) -> dict[str, Any]:
        model_params = chain(self.encoder.parameters(), self.decoder.parameters())
        optimizer = self.config.optimizer.constructor(model_params, **self.config.optimizer.kwargs)

        if len(self.config.schedulers) == 1:
            scheduler = self.config.schedulers[0]
            scheduler.kwargs["optimizer"] = optimizer
            scheduler = scheduler.build()

        else:
            schedulers = list()
            for scheduler in self.config.schedulers:
                scheduler.kwargs["optimizer"] = optimizer
                schedulers.append(scheduler.build())
            self.config.schedulers = schedulers

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimizer, 
                schedulers=self.config.schedulers,
                milestones=self.config.scheduler_milestones
            )        

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler} | self.config.scheduler_kwargs
        }


        #def get_lr_scheduler():
            #return self.lr_scheduler_constructor(config["optimizer"], **self.lr_scheduler_params)

        #def get_warmup_scheduler():
            #return self.warmup_scheduler_constructor(config["optimizer"], **self.warmup_scheduler_params)

        #def get_warmup_and_lr_scheduler():
            #return torch.optim.lr_scheduler.SequentialLR(config["optimizer"], [get_warmup_scheduler(), get_lr_scheduler()], [self.warmup_steps])

        #config = {"optimizer": self.optimizer_constructor(), **self.optimizer_params)}

        #if self.lr_scheduler_constructor is not None:
            #if self.warmup_scheduler_constructor is not None:
                #config["lr_scheduler"] = {"scheduler": get_warmup_and_lr_scheduler()} | self.scheduler_config_params 
            #else:
                #config["lr_scheduler"] = {"scheduler": get_lr_scheduler()} | self.scheduler_config_params 

        #elif self.warmup_scheduler_constructor is not None:
            #config["lr_scheduler"] = {"scheduler": get_warmup_scheduler()} | self.scheduler_config_params 

        #return config
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))
    
    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: batch_size: N, num_channels: C, height: H, width: W, num_classes: C'
        # NOTE: Classification: (NCHW):float -> model -> (NC'):float -> argmax(1) -> (N,):int
        # NOTE: Segmentation: (NCHW):float -> model -> (NC'HW):float -> argmax(1) -> (NHW):int
        images, labels = batch[0], batch[1]
        preds = self.forward(images) 
        loss = self.criterion(preds, labels)
        return preds, labels, loss

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch)
        if not self.trainer.validating and not self.trainer.sanity_checking:
            self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "preds": preds}
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._forward(batch)
        if not self.trainer.sanity_checking:
            self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "preds": preds}

    def test_step(self, batch, batch_idx):
        preds, _, loss = self._forward(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss, "preds": preds}

class SegmentationModule(lightning.LightningModule): ...
    # elif "UNet" in self.model_config.decoder_constructor.__qualname__:
        # self.model_config.decoder_params["layer_ch"] = self.encoder._out_ch_per_layer 
        # self.model_config.decoder_params["layer_up"] = self.encoder._downsampling_per_layer
        # self.forward = self._unet_forward

    # def _unet_forward(self, images: torch.Tensor) -> torch.Tensor:
        # encoder_outputs = list() 
        # for layer in self.encoder.children():
            # images = layer(images)
            # encoder_outputs.append(images)
        # return self.decoder(*reversed(encoder_outputs))



class DINO(lightning.LightningModule): ...

class MAE(lightning.LightningModule): ...

class Pix2Pix(lightning.LightningModule): ...

# class ModelConfig:
    # def __init__(
            # self,
            # encoder: str, 
            # encoder_params: dict[str, Any],
            # decoder: str,
            # decoder_params: dict[str, Any],
            # ckpt_path: Optional[str] = None
        # ):
        # self.encoder_constructor = self._get_constructor(encoder)
        # self.decoder_constructor = self._get_constructor(decoder)

        # assert isinstance(encoder_params, dict), f"config error (invalid type), expected :encoder_params to be dict, got {type(encoder_params)}"
        # for k in encoder_params.keys():
            # if k.endswith("_block"):
                # encoder_params[k] = getattr(blocks, encoder_params[k])
            # if k == "weights":
                # encoder_params[k] = self._get_weights_enum(encoder_params[k])
        # self.encoder_params = encoder_params 

        # assert isinstance(decoder_params, dict), f"config error (invalid type), expected :decoder_params to be dict, got {type(decoder_params)}"
        # for k in decoder_params.keys():
            # if k.endswith("_block"):
                # decoder_params[k] = getattr(blocks, decoder_params[k])
        # self.decoder_params = decoder_params

        # if ckpt_path is not None:
            # self.ckpt_path = fs.get_valid_file_err(ckpt_path)
        # else:
            # self.ckpt_path = None
    
    # def _get_weights_enum(self, name: str):
        # name: list = name.split('.')
        # get_weight: callable = getattr(import_module('.'.join(name[:2])), "get_weight")
        # return get_weight('.'.join(name[-2:]))

    # def _get_constructor(self, name: str) -> torch.nn.Module:
        # name = name.split('.') 
        # return getattr(import_module('.'.join(name[:-1])), name[-1])
