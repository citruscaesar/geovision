from typing import Literal, Optional
import torch
import torchvision # type: ignore

def alexnet(
        num_classes: int = 1000,
        dropout: float = 0.5,
        weights: Optional[Literal["imagenet", "he"]] = None,
        **kwargs 
    ) -> torch.nn.Module:

    match weights:
        case "imagenet":
            model = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
            if num_classes != 1000:
                model.classifier[-1] = torch.nn.Linear(4096, num_classes)
            return model
        case _:
            return torchvision.models.alexnet(num_classes = num_classes, dropout = dropout)

def resnet(
        num_layers: int = 18,
        num_classes: int = 1000,
        weights: Optional[Literal["imagenet", "he"]] = None,
        **kwargs
    ) -> torch.nn.Module:
    match num_layers:
        case 18:
            _resnet, _weights = torchvision.models.resnet18, torchvision.models.ResNet18_Weights
        case _:
            raise ValueError("invalid num_layers for resnet")
    
    match weights:
        case "imagenet":
            # TODO: how to init fc?
            model = _resnet(weights = _weights)
            #model.fc = torch.nn.Linear(512 * model.block.expansion, num_classes)
            return model
        case _:
            return _resnet(num_classes = num_classes)