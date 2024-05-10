import torch
import torchvision

def get_model_from_key(key: str) -> torch.nn.Module:
    match key:
        case "alexnet":
            torchvision.models.alexnet()