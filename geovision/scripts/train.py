import yaml
import logging
from dotenv import load_dotenv
from lightning import Trainer
from torchvision.transforms import v2 as T  # type: ignore
from geovision.io.local import get_new_dir, get_ckpt_path
from geovision.config.config import ExperimentConfig
from geovision.data.module import ImageDatasetDataModule
from models.module import ClassificationModule
from geovision.logging.loggers import get_csv_logger, get_ckpt_logger, get_classification_logger

if __name__ == "__main__":
    load_dotenv()
    with open("config.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
        config_dict["dataset_params"]["random_seed"] = config_dict["random_seed"]

    exec(config_dict["transforms_script"])
    config_dict["dataset_params"]["image_pre"] = image_pre  # type: ignore # noqa: F821
    config_dict["dataset_params"]["target_pre"] = target_pre  # type: ignore # noqa: F821
    config_dict["dataset_params"]["train_aug"] = train_aug  # type: ignore # noqa: F821
    config_dict["dataset_params"]["eval_aug"] = eval_aug  # type: ignore # noqa: F821

    config = ExperimentConfig.model_validate(config_dict)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f"{get_new_dir("logs") / config.name}.log",
        filemode="a",
        format="%(asctime)s : %(name)s : %(levelname)s : %(message)s",
        level=logging.INFO,
    )

    logger = [csv_logger:=get_csv_logger(config)]
    callbacks = list()
    callbacks.append(metrics_logger:=get_classification_logger(config))
    if config.trainer_params["enable_checkpointing"]:
        callbacks.append(ckpt_logger:=get_ckpt_logger(config))

    trainer = Trainer(logger=logger, callbacks=callbacks, **config.trainer_params)

    if config.trainer_task == "fit":
        trainer_fn = trainer.fit
    elif config.trainer_task == "validate":
        trainer_fn = trainer.validate
    elif config.trainer_task == "test":
        trainer_fn = trainer.test
    else:
        raise ValueError(
            f"trainer_task must be one of (fit, validate, test), got {config.trainer_task}"
        )

    trainer_fn(
        model=ClassificationModule(config),
        datamodule=ImageDatasetDataModule(config),
        ckpt_path=get_ckpt_path(config),
    )
