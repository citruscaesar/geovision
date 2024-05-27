from typing import Callable, Any, Optional
import torch
import torchvision # type: ignore
import torchmetrics
#from torchvision.transforms import v2 as T # type: ignore
from geovision.data.dataset import Dataset
from geovision.data.imagenette import ImagenetteImagefolderClassification
from geovision.data.imagenette import ImagenetteHDF5Classification 
from geovision.data.imagenet import ImagenetImagefolderClassification
from geovision.data.imagenet import ImagenetHDF5Classification
from geovision.models import alexnet
from geovision.models import resnet 

def get_dataset(name: str) -> Dataset:
    datasets = {
        "imagenette_imagefolder_classification": ImagenetteImagefolderClassification,
        "imagenette_hdf5_classification": ImagenetteHDF5Classification,
        "imagenet_imagefolder_classification": ImagenetImagefolderClassification,
        "imagenet_hdf5_classification": ImagenetHDF5Classification 
    }
    return get_constructor_err(datasets, name) # type: ignore

def get_task(dataset: Dataset) -> str:
    task_str = dataset.name.split('_')[-1]
    match task_str:
        case "classification":
            return "multiclass" if dataset.num_classes > 2 else "binary"
        case "multilabelclassification":
            return "multilabel"
        case _:
            raise ValueError(f"{dataset} has invalid task str, got {task_str}")

def get_model(name: str) -> torch.nn.Module:
    models = {
        "alexnet": alexnet,
        "resnet": resnet,
    }
    return get_constructor_err(models, name)

def get_metric(name: str) -> torchmetrics.Metric:
    metrics = {
        "accuracy": torchmetrics.Accuracy,
        "f1": torchmetrics.F1Score,
        "iou": torchmetrics.JaccardIndex,
        "confusion_matrix": torchmetrics.ConfusionMatrix,
        "cohen_kappa": torchmetrics.CohenKappa,
        "auroc": torchmetrics.AUROC,
    }
    return get_constructor_err(metrics, name)

def get_criterion(name: str) -> torch.nn.Module:
    criterions = {
        "binary_cross_entropy": torch.nn.BCEWithLogitsLoss,
        "cross_entropy": torch.nn.CrossEntropyLoss,
        "mean_squared_error": torch.nn.MSELoss,
    }
    return get_constructor_err(criterions, name) # type: ignore
    
def get_optimizer(name: str) -> torch.optim.Optimizer:
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop
    }
    return get_constructor_err(optimizers, name) # type: ignore

def get_constructor_err(fn_table: dict["str", Any], name: str) -> Any:
    if name not in fn_table:
        raise NotImplementedError(f"{name} must be one of {fn_table.keys()}")
    else:
        return fn_table[name]

# class TransformParser:
    # ref = {
        # "to_image": T.ToImage(),
        # "to_float32_scaled": T.ToDtype(torch.float32, scale = True),
        # "to_float32_unscaled": T.ToDtype(torch.float32, scale = False),
    # }

    # @classmethod
    # def parse(cls, transforms: list[str]):
        # parsed_transforms: list[T.Transform]
        # for transform in transforms[1:]:
            # pass

# # transform: [composition, t1, t2, t3, ...]
# # apply_composition: str, list -> composed

    # @classmethod
    # def compose(cls, composition: str, transforms: list[T.Transform]):
        # p_str = composition.split('_')[-1]
        # if p_str != '':
            # p = float(p_str) / 100
            # if "random_apply" in composition:
                # return T.RandomApply(transforms, p) 
            # elif "random_choice" in composition:
                # return T.RandomChoice(transforms, p)
        # elif composition == "compose":
            # return T.Compose(transforms)
        # elif composition == "random_order":
            # return T.RandomOrder(transforms)
        # else:
            # raise ValueError(
                # f""":key should be one of [compose, random_order, random_apply_p, random_choice_p], 
                # got {composition}"""
            # )