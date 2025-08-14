import torch
import logging
import warnings
import argparse
from dotenv import load_dotenv
from lightning import Trainer
from geovision.io.local import FileSystemIO as fs
from lightning.pytorch.profilers import PyTorchProfiler
from geovision.experiment.config import ExperimentConfig
from geovision.data import ImageDatasetDataModule
from geovision.models.interfaces import ClassificationModule
from geovision.experiment.loggers import get_csv_logger, get_ckpt_logger, get_classification_logger

if __name__ == "__main__":
    load_dotenv()
    cli = argparse.ArgumentParser("training and evaluation script for geovision")
    cli.add_argument("--mode", choices=["fit", "validate", "test"], required=True)
    cli.add_argument("--config", default="config.yaml", help = "path to config.yaml")
    cli.add_argument("--compile", action="store_true", help = "optimize with torch.compile")
    cli.add_argument("--profile", action="store_true", help = "init profiler")
    cli.add_argument("--reset_logs", action="store_true", help = "reset experiment logs directory (will also delete saved models)")
    args = cli.parse_args()
    print(args)

    config = ExperimentConfig.from_yaml(args.config)

    logging.basicConfig(
        filename = f"{fs.get_new_dir(config.experiments_dir, "logs")}/logfile.log",
        filemode = "a",
        format = "%(asctime)s : %(name)s : %(levelname)s : %(message)s",
        level = logging.INFO
    )
    def log_warnings(message, category, filename, lineno, file=None, line=None):
        logging.info(f"{filename} : {lineno} : {category.__name__} : {message}")
    warnings.showwarning = log_warnings

    if args.reset_logs:
        import shutil
        shutil.rmtree(config.experiments_dir)
        config.experiments_dir.mkdir()

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

    if args.compile:
        model = torch.compile(model, fullgraph = True)

    if args.profile:
        config.trainer_params["profiler"] = PyTorchProfiler(dirpath=config.experiments_dir/"logs", filename="profiler", export_to_chrome=True)

    trainer = Trainer(logger=loggers, callbacks=callbacks, **config.trainer_params)

    if args.mode == "fit":
        run = trainer.fit
    elif args.mode == "validate":
        run = trainer.validate
    elif args.mode == "test":
        run = trainer.test
    else:
        raise ValueError(f":trainer_task must be one of (fit, validate, test), got {config.trainer_task}")

    run(
        model = model,
        datamodule = ImageDatasetDataModule(config.dataset_constructor, config.dataset_config, config.dataloader_config),
        ckpt_path = config.ckpt_path,
    )
