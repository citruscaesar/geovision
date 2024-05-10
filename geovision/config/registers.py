from typing import Callable, Any
import torch
import torchvision
import torchmetrics
#from torchvision.transforms import v2 as T # type: ignore
from geovision.data.imagenette import ImagenetteImagefolderClassification
from geovision.data.imagenette import ImagenetteHDF5Classification 

def get_metric(name: str, init_params: dict[str, Any]) -> Callable:
    metrics = {
        "accuracy": torchmetrics.Accuracy,
        "f1": torchmetrics.F1Score,
        "iou": torchmetrics.JaccardIndex,
        "confusion_matrix": torchmetrics.ConfusionMatrix,
        "cohen_kappa": torchmetrics.CohenKappa,
        "auroc": torchmetrics.AUROC,
    }
    return _get_fn_from_table(metrics, name)(**init_params)

def get_criterion(name: str, init_params: dict[str, Any]) -> Callable:
    criterions = {
        "binary_cross_entropy": torch.nn.BCEWithLogitsLoss,
        "cross_entropy": torch.nn.CrossEntropyLoss,
        "mean_squared_error": torch.nn.MSELoss,
    }
    return _get_fn_from_table(criterions, name)(**init_params)
    
def get_optimizer(name: str, init_params: dict[str, Any]) -> Callable:
    optimizers = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }
    return _get_fn_from_table(optimizers, name)(**init_params)

def get_dataset(name: str) -> Callable:
    datasets = {
        "imagenette_imagefolder_classification": ImagenetteImagefolderClassification,
        "imagenette_hdf5_classification": ImagenetteHDF5Classification 
    }
    return _get_fn_from_table(datasets, name)

def get_model(name: str, init_params: dict[str, Any]):
    models = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34
    }
    return _get_fn_from_table(models, name)(**init_params)

def _get_fn_from_table(fn_table: dict["str", Callable], name: str) -> Callable:
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