import logging
import warnings
from dotenv import load_dotenv
from lightning import Trainer
#from torchvision.transforms import v2 as T  # type: ignore
from geovision.io.local import FileSystemIO as fs
from lightning.pytorch.profilers import PyTorchProfiler
from geovision.experiment.config import ExperimentConfig
from geovision.data import ImageDatasetDataModule
from geovision.models.interfaces import ClassificationModule
from geovision.experiment.loggers import get_csv_logger, get_ckpt_logger, get_classification_logger

# 1. Add options to change mode [fit, validate, test], select config.yaml, select profiling, delete logs dir before starting, to run.py
# 2. Redirect [all] warnings to logfile, not to stdout

def log_warnings(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{filename} : {lineno} : {category.__name__}: {message}")
warnings.showwarning = log_warnings

if __name__ == "__main__":
    load_dotenv()
    config = ExperimentConfig.from_yaml("config.yaml")

    logging.basicConfig(
        filename = f"{fs.get_new_dir(config.experiments_dir, "logs")}/logfile.log",
        filemode = "a",
        format = "%(asctime)s : %(name)s : %(levelname)s : %(message)s",
        level = logging.INFO
    )

    loggers: list = list()
    loggers.append(csv_logger := get_csv_logger(config))

    callbacks: list = list()
    callbacks.append(metrics_logger := get_classification_logger(config))
    if config.trainer_params["enable_checkpointing"]:
        callbacks.append(ckpt_logger:=get_ckpt_logger(config))

    model = ClassificationModule(
        model_config=config.model_config,
        criterion_constructor=config.criterion_constructor, 
        criterion_params=config.criterion_params, 
        optimizer_constructor=config.optimizer_constructor,
        optimizer_params=config.optimizer_params,
        lr_scheduler_constructor=config.scheduler_constructor,
        lr_scheduler_params=config.scheduler_params,
        warmup_scheduler_constructor=config.warmup_scheduler_constructor,
        warmup_scheduler_params=config.warmup_scheduler_params,
        warmup_steps=config.warmup_steps,
        scheduler_config_params=config.scheduler_config_params
    )

    profiler = PyTorchProfiler(
        dirpath=config.experiments_dir / "logs", 
        filename="profiler",
        export_to_chrome=True,
    )

    trainer = Trainer(logger=loggers, callbacks=callbacks, profiler=profiler, **config.trainer_params)
    if config.trainer_task == "fit":
        run = trainer.fit
    elif config.trainer_task == "validate":
        run = trainer.validate
    elif config.trainer_task == "test":
        run = trainer.test
    else:
        raise ValueError(f":trainer_task must be one of (fit, validate, test), got {config.trainer_task}")

    run(
        model = model,
        datamodule = ImageDatasetDataModule(config.dataset_constructor, config.dataset_config, config.dataloader_config),
        ckpt_path = config.ckpt_path,
    )
