from typing import Literal

import torch
import torchvision

def get_alexnet(
        weights: Literal["imagenet", "he", "gaussian", "uniform"] = "he",
        num_classes: int = 1000,
        dropout: float = 0.5,
        **kwargs 
    ) -> torch.nn.Module:
    match weights:
        case "imagenet":
            model = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
            model.classifier[-1] = torch.nn.Linear(4096, num_classes, True)
            model.classifier[-1].weight = torch.nn.init.kaiming_normal_(model.classifier[-1].weight, nonlinearity="relu")
        case "he":
            model = torchvision.models.alexnet(num_classes = num_classes, dropout = dropout)
            model.weight = torch.nn.init.kaiming_normal_(model.weight, nonlinearity="relu")
        case "gaussian":
            model = torchvision.models.alexnet(num_classes = num_classes, dropout = dropout)
            model.weight = torch.nn.init.normal_(model.weight, nonlinearity="relu")
        case "random":
            model = torchvision.models.alexnet(num_classes = num_classes, dropout = dropout)
        case _:
            raise ValueError("invalid :weights")
    return model    

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes:int = 1000, dropout:int = 0.5):
        self.classifier = torch.nn.Sequential([
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            torch.nn.LocalResponseNorm()
        ])