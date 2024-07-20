from typing import Any, Callable
import torch
import torchmetrics
#from torchvision.transforms import v2 as T # type: ignore
from geovision.data.dataset import Dataset

from geovision.models import alexnet
from geovision.models import resnet 

_models = {
    "alexnet": alexnet,
    "resnet": resnet,
}

_metrics = {
    "accuracy": torchmetrics.Accuracy,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
    "f1": torchmetrics.F1Score,
    "iou": torchmetrics.JaccardIndex,
    "confusion_matrix": torchmetrics.ConfusionMatrix,
    "cohen_kappa": torchmetrics.CohenKappa,
    "auroc": torchmetrics.AUROC,
}

_criterions = {
    "binary_cross_entropy": torch.nn.BCEWithLogitsLoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "mean_squared_error": torch.nn.MSELoss,
}

_optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop
}

_schedulers = {
    "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

def _validate_key_err(fn_table: dict[str, Any], name: str):
    if name not in fn_table.keys():
        raise NotImplementedError(f"{name} must be one of {fn_table.keys()}")

def validate_dataset_err(name: str):
    _datasets = [
        "imagenette_imagefolder_classification",
        "imagenette_hdf5_classification",
        "imagenet_imagefolder_classification",
        "imagenet_hdf5_classification",
    ]
    if name not in _datasets:
        raise NotImplementedError(f"{name} must be one of {_datasets}")

def validate_model_err(name: str):
    _validate_key_err(_models, name)

def validate_metric_err(name: str):
    _validate_key_err(_metrics, name)

def validate_criterion_err(name: str):
    _validate_key_err(_criterions, name)

def validate_optimizer_err(name: str):
    _validate_key_err(_optimizers, name)

def validate_scheduler_err(name: str):
    _validate_key_err(_schedulers, name)

def parse_dataset(name: str) -> Callable:
    match name:
        case "imagenette_imagefolder_classification":
            from geovision.data.imagenette import ImagenetteImagefolderClassification
            return ImagenetteImagefolderClassification
        case "imagenette_hdf5_classification":
            from geovision.data.imagenette import ImagenetteHDF5Classification 
            return ImagenetteHDF5Classification
        case "imagenet_imagefolder_classification":
            from geovision.data.imagenet import ImagenetImagefolderClassification
            return ImagenetImagefolderClassification
        case "imagenet_hdf5_classification":
            from geovision.data.imagenet import ImagenetHDF5Classification
            return ImagenetHDF5Classification 
        case _:
            return 

def parse_model(name: str) -> Callable:
    return _models[name] 

def parse_metric(name: str) -> Callable:
    return _metrics[name] 

def parse_criterion(name: str) -> Callable:
    return _criterions[name] 

def parse_optimizer(name: str) -> Callable:
    return _optimizers[name] 

def parse_scheduler(name: str) -> Callable:
    return _schedulers[name]

def parse_task(dataset: Dataset) -> str:
    task_str = dataset.name.split('_')[-1]
    match task_str:
        case "classification":
            return "multiclass" if dataset.num_classes > 2 else "binary"
        case "multilabelclassification":
            return "multilabel"
        case _:
            raise ValueError(f"{dataset} has invalid task str, got {task_str}")