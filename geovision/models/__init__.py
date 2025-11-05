from typing import Literal, Optional

import torch
from geovision.io.local import FileSystemIO as fs

def get_state_dict(weights_init: Literal["torchvision", "torchgeo", "url", "path"], weights_param: Optional[str] = None):
    valid_weights_inits = ("torchvision", "torchgeo", "timm", "url", "path")
    assert weights_init in valid_weights_inits, f"config error, expected :weights to be one of {valid_weights_inits}, got {weights_init}"

    if weights_init in ("torchvision", "torchgeo"):
        assert weights_param is not None, f"config error, did not expect :weights_param be None when :weights_init is {weights_init}"
        if weights_init == "torchvision":
            from torchvison.models import get_weight
            weights = get_weight(weights_param)
        elif weights_init == "torchgeo":
            from torchgeo.models import get_weight
            weights = get_weight(weights_param)
        return weights.get_state_dict()

    elif weights_init == "path":
        weights = fs.get_valid_file_err(weights_param)
        return torch.load(weights, weights_only=True)

    elif weights_init == "url":
        assert isinstance(weights_param, str), f"config error, expected :weights_param to be a valid url when :weights is url, got {weights_param}"
        raise NotImplementedError("loading weights from random URLs is not implemented yet")

# Export ResNet models
from geovision.models.resnet import ResNet_Encoder

# Export ViT models
from geovision.models.vit import (
    VisionTransformer,
    ViTEncoder,
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_large_patch16_224,
    vit_huge_patch14_224,
)

# Export ConvNeXt models
from geovision.models.convnext import (
    ConvNeXt_Encoder,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    convnext_xlarge,
    convnextv2_tiny,
    convnextv2_small,
    convnextv2_base,
    convnextv2_large,
    convnextv2_xlarge,
)

# Export Swin Transformer models
from geovision.models.swin import (
    SwinTransformer_Encoder,
    swin_tiny,
    swin_small,
    swin_base,
    swin_large,
    swinv2_tiny,
    swinv2_small,
    swinv2_base,
    swinv2_large,
)

__all__ = [
    # Utilities
    "get_state_dict",
    # ResNet
    "ResNet_Encoder",
    # ViT
    "VisionTransformer",
    "ViTEncoder",
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "vit_huge_patch14_224",
    # ConvNeXt
    "ConvNeXt_Encoder",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
    "convnextv2_tiny",
    "convnextv2_small",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_xlarge",
    # Swin Transformer
    "SwinTransformer_Encoder",
    "swin_tiny",
    "swin_small",
    "swin_base",
    "swin_large",
    "swinv2_tiny",
    "swinv2_small",
    "swinv2_base",
    "swinv2_large",
]