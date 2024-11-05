from typing import Callable, Optional

import yaml
import torch
import logging
import torchmetrics
from pathlib import Path

from geovision.data.interfaces import DatasetConfig, DataLoaderConfig
from geovision.io.local import FileSystemIO as fs

logger = logging.getLogger(__name__)

class ExperimentConfig:
    def __init__(
        self,
        project_name: str,
        trainer_task: str,
        random_seed: str,
        dataset_name: str,
        dataset_params: dict,
        dataloader_params: dict,
        model_name: str,
        model_params: dict,
        log_params: dict,
        trainer_params: dict,
        metric_name: str,
        metric_params: Optional[dict] = None,
        run_name: Optional[str | int] = None,
        criterion_name: Optional[str] = None,
        criterion_params: Optional[dict] = None,
        optimizer_name: Optional[str] = None,
        optimizer_params: Optional[dict] = None,
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        scheduler_config_params: Optional[dict] = None,
        warmup_scheduler_name: Optional[str] = None,
        warmup_scheduler_params: Optional[dict] = None,
        warmup_steps: Optional[int] = None,
        **kwargs,
    ):
        assert project_name is not None, "config error (missing value), :project_name cannot be empty"
        assert isinstance(project_name, str), f"config error (invalid type), expected :project_name to be str, got {type(project_name)}"
        self.project_name = str(project_name)

        if run_name is None or run_name == "":
            from names_generator import generate_name

            self.run_name = generate_name()
            warning_str = f"config warning (missing value), using generated :run_name '{self.run_name}'"
            print(warning_str)
            logger.info(warning_str)
        else:
            assert isinstance(run_name, str) or isinstance(run_name, int), f"config error (invalid type), expected :run_name to be str or int, got {type(run_name)}"
            self.run_name = str(run_name)

        assert trainer_task in ("fit", "validate", "test"), f"config error (invalid value), expected :trainer_task to be one of fit, validate, test or predict, got {trainer_task}"
        assert isinstance(trainer_task, str), f"config error (invalid type), expected :trainer_task to be str, got {type(trainer_task)}"
        self.trainer_task = trainer_task

        assert isinstance(random_seed, int), f"config error (invalid type), expected :random_seed to be int, got {type(random_seed)}"
        self.random_seed = random_seed

        assert isinstance(dataset_name, str), f"config error (invalid type), expected :dataset_name to be str, got {type(dataset_name)}"
        self.dataset_constructor = self._get_dataset_constructor(dataset_name)  # NOTE: also checks if dataset_name is valid; call this before setting self.dataset_name
        self.dataset_name = dataset_name

        assert isinstance(dataset_params, dict), f"config error (invalid type), expected :dataset_params to be dict, got {type(dataset_params)}"
        self.dataset_config = DatasetConfig(random_seed=self.random_seed, **dataset_params)

        assert isinstance(dataloader_params, dict), f"config error (invalid type), expected :dataloader_params to be dict, got {type(dataloader_params)}"

        dataloader_params["batch_transform_name"] = dataset_params.get("batch_transform_name", None)
        if dataloader_params["batch_transform_name"] is not None:
            if dataset_params["batch_transform_params"] is not None:
                assert isinstance(
                    dataset_params["batch_transform_params"], dict
                ), f"config error (invalid type), expected :batch_transform-params to be dict, got {type(dataset_params["batch_transform_params"])}"
            dataloader_params["batch_transform_params"] = (dataset_params["batch_transform_params"] or dict()) | {"num_classes": self.dataset_constructor.num_classes}
        self.dataloader_config = DataLoaderConfig(**dataloader_params)

        assert isinstance(model_name, str), f"config error (invalid type), expected :model_name to be str, got {type(model_name)}"
        assert isinstance(model_params, dict), f"config error (invalid type), expected :model_params to be dict, got {type(model_params)}"
        self.model_name = model_name
        self.model_params = model_params | {"num_classes": self.dataset_constructor.num_classes}
        self.model_constructor = self._get_model_constructor(model_name)  # also checks if model_name is valid

        if criterion_name is not None:
            assert isinstance(criterion_name, str), f"config error (invalid type), expected :criterion_name to be str, got {type(criterion_name)}"
            self.criterion_constructor = self._get_criterion_constructor(criterion_name)  # also checks if criterion_name is valid
            if criterion_params is None:
                self.criterion_params = dict()
            else:
                assert isinstance(criterion_params, dict), f"config error (invalid type), expected :criterion_params to be dict, got {type(criterion_params)}"
                self.criterion_params = criterion_params
        self.criterion_name = criterion_name

        if metric_name is not None:
            assert isinstance(metric_name, str), f"config error (invalid type), expected :metric_name to be str, got {type(metric_name)}"
            self.metric_constructor = self._get_metric_constructor(metric_name)  # also checks if metric_name is valid
            if metric_params is None:
                self.metric_params = dict()
            else:
                assert isinstance(metric_params, dict), f"config error (invalid type), expected :metric_params to be dict, got {type(metric_params)}"
                self.metric_params = metric_params
            self.metric_params.update({"task": self._get_metric_task(self.dataset_name), "num_classes": self.dataset_constructor.num_classes})
        self.metric_name = metric_name

        if optimizer_name is not None:
            assert isinstance(optimizer_name, str), f"config error (invalid type), expected :optimizer_name to be str, got {type(optimizer_name)}"
            self.optimizer_constructor = self._get_optimizer_constructor(optimizer_name)  # also checks if optimizer_name is valid
            if optimizer_params is None:
                self.optimizer_params = dict()
            else:
                assert isinstance(optimizer_params, dict), f"config error (invalid type), expected :optimizer_params to be dict, got {type(optimizer_params)}"
                self.optimizer_params = optimizer_params
        self.optimizer_name = optimizer_name

        if scheduler_name is not None:
            assert isinstance(scheduler_name, str), f"config error (invalid type), expected :scheduler_name to be str, got {type(scheduler_name)}"
            self.scheduler_constructor = self._get_scheduler_constructor(scheduler_name)  # also checks if scheduler_name is valid
        self.scheduler_name = scheduler_name

        if scheduler_params is None:
            self.scheduler_params = dict()
        else:
            assert isinstance(scheduler_params, dict), f"config error (invalid type), expected :scheduler_params to be dict, got {type(scheduler_params)}"
            self.scheduler_params = scheduler_params

        if warmup_scheduler_name is not None:
            assert isinstance(warmup_scheduler_name, str), f"config error (invalid type), expected :warmup_scheduler_name to be str, got {type(warmup_scheduler_name)}"
            assert isinstance(warmup_steps, int), f"config error (invalid type), expected :warmup_steps to be int, got {type(warmup_steps)}"
            self.warmup_scheduler_constructor = self._get_scheduler_constructor(warmup_scheduler_name)  # also checks if scheduler_name is valid
        self.warmup_scheduler_name = warmup_scheduler_name
        self.warmup_steps = warmup_steps

        if warmup_scheduler_params is None:
            self.warmup_scheduler_params = dict()
        else:
            assert isinstance(warmup_scheduler_params, dict), f"config error (invalid type), expected :warmup_scheduler_params to be dict, got {type(warmup_scheduler_params)}"
            self.warmup_scheduler_params = warmup_scheduler_params

        if scheduler_config_params is None:
            self.scheduler_config_params = dict()
        else:
            assert isinstance(scheduler_config_params, dict), f"config error (invalid type), expected :scheduler_config_params to be dict, got {type(scheduler_config_params)}"
            self.scheduler_config_params = scheduler_config_params

        assert isinstance(log_params, dict), f"config error (invalid type), expected :log_params to be dict, got {type(log_params)}"
        log_every_n_steps = log_params.get("log_every_n_steps")
        assert log_every_n_steps is not None, "config error (missing value), expected :log_params to contain log_every_n_steps(int)"
        assert isinstance(log_every_n_steps, int), f"config error (invalid value), expected :log_params[log_every_n_steps] to be an int, got {type(log_every_n_steps)}"

        log_every_n_epochs = log_params.get("log_every_n_epochs")
        assert log_every_n_epochs is not None, "config error (missing value), expected :log_params to contain log_every_n_epochs(int)"
        assert isinstance(log_every_n_epochs, int), f"config error (invalid value), expected :log_params[log_every_n_epochs] to be an int, got {type(log_every_n_epochs)}"

        log_model_outputs = log_params.get("log_model_outputs")
        assert log_model_outputs is not None, "config error (missing value), expected :log_params to contain log_model_outputs(int)"
        assert isinstance(log_model_outputs, int), f"config error (invalid value), expected :log_params[log_model_outputs] to be an int, got {type(log_model_outputs)}"
        assert (
            log_model_outputs >= -1
        ), "config error (invalid value), expected :log_params[log_model_outputs] to be 0 for no logging, -1 to log all outputs or any k < num_dataset_classes to log top_k"
        assert log_model_outputs <= self.dataset_constructor.num_classes

        assert log_params.get("log_to_h5") is not None, "config error (missing value), expected :log_params to contain log_to_h5(bool)"
        assert log_params.get("log_to_wandb") is not None, "config error (missing value), expected :log_params to contain log_to_wandb(bool)"
        assert log_params.get("log_to_csv") is not None, "config error (missing value), expected :log_params to contain log_to_csv(bool)"
        self.log_params = log_params

        assert isinstance(trainer_params, dict), f"config error (invalid type), expected :trainer_params to be dict, got {type(trainer_params)}"
        trainer_params["log_every_n_steps"] = self.log_params["log_every_n_steps"]
        trainer_params["check_val_every_n_epoch"] = self.log_params["log_every_n_epochs"]
        trainer_params["accumulate_grad_batches"] = self.grad_accum
        self.trainer_params = trainer_params

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        namespace = dict()
        exec(config_dict["dataset_params"]["transforms"], namespace)
        config_dict["dataset_params"]["image_pre"] = namespace["image_pre"]  # type: ignore # noqa: F821
        config_dict["dataset_params"]["target_pre"] = namespace["target_pre"]  # type: ignore # noqa: F821
        config_dict["dataset_params"]["train_aug"] = namespace["train_aug"]  # type: ignore # noqa: F821
        config_dict["dataset_params"]["eval_aug"] = namespace["eval_aug"]  # type: ignore # noqa: F821
        return cls(**config_dict)

    def __repr__(self) -> str:
        out = f"==Experiment Config==\nProject Name: {self.project_name}\nRun Name: {self.run_name}\nRandom Seed: {self.random_seed}\n\n"
        out += f"==Dataset Config==\nDataset: {self.dataset_name} [{self.dataset_constructor}]\n{self.dataset_params}Dataloader Params: {self.dataloader_config}\n\n"
        out += "==Model Config==\n\n"
        out += f"==Task Config==\nTrainer Task: {self.trainer_task}\nTrainer Params: {self.trainer_params}\n\n"
        out += f"==Evaluation Config==\nCriterion: {self.criterion_name} [{self.criterion_constructor}]\nCriterion Params: {self.criterion_params}\n"
        out += f"Metric: {self.metric_name} [{self.metric_constructor}]\nMetric Params: {self.metric_params}\n\n"
        out += f"==Training Config==\nOptimizer: {self.optimizer_name} [{self.optimizer_constructor}]\nOptimizer Params: {self.optimizer_params}\n"
        out += f"LR Scheduler: {self.scheduler_name} [{self.scheduler_constructor}]\nLR Scheduler Params: {self.scheduler_params}\nLR Scheduler Config: {self.scheduler_config_params}\n\n"
        out += f"==Logging Config==\n{"\n".join(str(self.log_params).removeprefix("{").removesuffix("}").split(", "))}\n"
        return out

    @property
    def experiments_dir(self) -> Path:
        """returns path to (created if non-existant) experiments log dir, ~/experiments/{project_name}/{run_name}"""
        return fs.get_new_dir(Path.home(), "experiments", self.project_name, self.run_name)

    @property
    def last_ckpt_path(self, epoch: int = -1, step: int = -1) -> Path:
        def display_ckpts_list(ckpts) -> None:
            names = [ckpt.name for ckpt in ckpts]
            names[-1] = f"*{names[-1]}"
            display(f"found ckpts: {names}") # noqa # type: ignore

        if epoch < -1:
            raise ValueError(f"epoch must be >= -1, got{epoch}")
        if step < -1:
            raise ValueError(f"step must be >= -1, got {step}")

        # dosen't sort properly -> gather epoch_idx and step_idx separately and then sort 
        # try not to use pandas
        ckpts = {int(x.stem.split('=')[-1]):x for x in self.experiments_dir.rglob("*.ckpt")}
        ckpts = [ckpts[x] for x in sorted(ckpts)] 
        if len(ckpts) == 0:
            print("no ckpt found in experiments, returning None")
        elif epoch == -1 and step == -1:
            display_ckpts_list(ckpts)
            return ckpts[-1] 
        elif epoch != -1 and step == -1:
            ckpts = [ckpt for ckpt in ckpts if f"epoch={epoch}" in ckpt.name]
            display_ckpts_list(ckpts)
            return ckpts[-1]
        else:
            ckpts = [ckpt for ckpt in ckpts if f"epoch={epoch}_step={step}" in ckpt.name]
            if len(ckpts) == 0:
                print("found no matching ckpt, returning None")
                return None
            else:
                display_ckpts_list(ckpts)
                return ckpts[-1]

    @property
    def wandb_params(self) -> dict:
        return {"project": self.project_name, "name": self.run_name, "dir": self.experiments_dir, "config": self.__dict__, "resume": "never"} | self.log_params.get("wandb_params", dict())

    @property
    def grad_accum(self) -> int:
        return self.dataloader_config.gradient_accumulation

    def get_metric(self, metric_name: Optional[str] = None, addl_metric_params: Optional[dict] = None):
        metric_name: str = metric_name or self.metric_name
        metric_params: dict = self.metric_params 
        if addl_metric_params is not None:
            metric_params = metric_params | addl_metric_params
        return self._get_metric_constructor(metric_name)(**metric_params)

    def _get_metric_task(self, dataset_name: str) -> str:
        task_str = dataset_name.split("_")[-1]
        match task_str:
            case "classification":
                return "multiclass" if self.dataset_constructor.num_classes > 2 else "binary"
            case "multilabelclassification":
                return "multilabel"
            case _:
                raise AssertionError(f"config error (invalid value), :dataset_name has an invalid task str, got {task_str}")

    def _get_model_constructor(self, model_name) -> torch.nn.Module:
        match model_name:
            case "alexnet":
                from torchvision.models import alexnet
                return alexnet
            case "resnet":
                from geovision.models.resnet import ResNet
                return ResNet 
            case _:
                raise AssertionError(f"config error (not implemented), got {model_name}")

    def _get_metric_constructor(self, metric_name: str):
        match metric_name:
            case "accuracy":
                return torchmetrics.Accuracy
            case "precision":
                return torchmetrics.Precision
            case "recall":
                return torchmetrics.Recall
            case "f1":
                return torchmetrics.F1Score
            case "iou":
                return torchmetrics.JaccardIndex
            case "confusion_matrix":
                return torchmetrics.ConfusionMatrix
            case "cohen_kappa":
                return torchmetrics.CohenKappa
            case "auroc":
                return torchmetrics.AUROC
            case _:
                raise AssertionError(f"config error (not implemented), got {metric_name}")

    def _get_dataset_constructor(self, dataset_name: str):
        match dataset_name:
            case "imagenette_imagefolder_classification":
                from geovision.data.imagenet import ImagenetteImagefolderClassification

                return ImagenetteImagefolderClassification
            case "imagenette_hdf5_classification":
                from geovision.data.imagenet import ImagenetteHDF5Classification

                return ImagenetteHDF5Classification
            case "imagenet_imagefolder_classification":
                from geovision.data.imagenet import ImagenetImagefolderClassification

                return ImagenetImagefolderClassification
            case "imagenet_hdf5_classification":
                from geovision.data.imagenet import ImagenetHDF5Classification

                return ImagenetHDF5Classification
            case "imagenet_litdata_classification":
                from geovision.data.imagenet import ImagenetLitDataClassification

                return ImagenetLitDataClassification
            case "fmow_hdf5_classification":
                from geovision.data.fmow import FMoWHDF5Classification

                return FMoWHDF5Classification
            case _:
                raise AssertionError(f"config error (not implemented), got {dataset_name}")

    def _get_criterion_constructor(self, criterion_name: str) -> torch.nn.Module:
        match criterion_name:
            case "binary_cross_entropy_with_logits":
                return (torch.nn.BCEWithLogitsLoss,)
            case "l1":
                return torch.nn.L1Loss
            case "nll":
                return torch.nn.NLLLoss
            case "poisson_nll":
                return torch.nn.PoissonNLLLoss
            case "hinge":
                return torch.nn.HingeEmbeddingLoss
            case "multilabel_margin":
                return torch.nn.MultiLabelMarginLoss
            case "smooth_l1":
                return torch.nn.SmoothL1Loss
            case "huber":
                return torch.nn.HuberLoss
            case "soft_margin":
                return torch.nn.SoftMarginLoss
            case "cross_entropy":
                return torch.nn.CrossEntropyLoss
            case "multilabel_soft_margin":
                return torch.nn.MultiLabelSoftMarginLoss
            case "cosine_embedding":
                return torch.nn.CosineEmbeddingLoss
            case "margin_ranking":
                return torch.nn.MarginRankingLoss
            case "multi_margin":
                return torch.nn.MultiMarginLoss
            case "triplet_margin":
                return torch.nn.TripletMarginLoss
            case "triplet_margine_with_distance":
                return torch.nn.TripletMarginWithDistanceLoss
            case "ctc":
                return torch.nn.CTCLoss
            case _:
                raise AssertionError(f"config error (not implemented), got {criterion_name}")

    def _get_optimizer_constructor(self, optimizer_name: str) -> torch.optim.Optimizer:
        match optimizer_name:
            case "sgd":
                return torch.optim.SGD
            case "adam":
                return torch.optim.Adam
            case "adamw":
                return torch.optim.AdamW
            case "adadelta":
                return torch.optim.Adadelta
            case "adagrad":
                return torch.optim.Adagrad
            case "adamax":
                return torch.optim.Adamax
            case "asgd":
                return torch.optim.ASGD
            case "lbfgs":
                return torch.optim.LBFGS
            case "nadam":
                return torch.optim.NAdam
            case "radam":
                return torch.optim.RAdam
            case "rmsprop":
                return torch.optim.RMSprop
            case "rprop":
                return torch.optim.Rprop
            case "sparseadam":
                return torch.optim.SparseAdam
            case _:
                raise AssertionError(f"config error (not implemented), got {optimizer_name}")

    def _get_scheduler_constructor(self, scheduler_name: str) -> torch.optim.lr_scheduler.LRScheduler:
        match scheduler_name:
            case "reduce_lr_on_plateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau
            case "lambda_lr":
                return torch.optim.lr_scheduler.LambdaLR
            case "multiplicative_lr":
                return torch.optim.lr_scheduler.MultiplicativeLR
            case "step_lr":
                return torch.optim.lr_scheduler.StepLR
            case "multistep_lr":
                return torch.optim.lr_scheduler.MultiStepLR
            case "constant_lr":
                return torch.optim.lr_scheduler.ConstantLR
            case "linear_lr":
                return torch.optim.lr_scheduler.LinearLR
            case "exponential_lr":
                return torch.optim.lr_scheduler.ExponentialLR
            case "sequential_lr":
                return torch.optim.lr_scheduler.SequentialLR
            case "cosine_annealing_lr":
                return torch.optim.lr_scheduler.CosineAnnealingLR
            case "chained_scheduler_lr":
                return torch.optim.lr_scheduler.ChainedScheduler
            case "reduce_lr_on_plateau_lr":
                return torch.optim.lr_scheduler.ReduceLROnPlateau
            case "cyclic_lr":
                return torch.optim.lr_scheduler.CyclicLR
            case "cosine_annealing_warm_restarts_lr":
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            case "one_cycle_lr":
                return torch.optim.lr_scheduler.OneCycleLR
            case "polynomial_lr":
                return torch.optim.lr_scheduler.PolynomialLR
            case _:
                raise AssertionError(f"config error (not implemented), got {scheduler_name}")

if __name__ == "__main__":
    config = ExperimentConfig.from_config("geovision/scripts/new_config.yaml")
    print(config)
