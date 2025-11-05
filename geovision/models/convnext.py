from typing import Optional, Literal, Sequence
import math

import torch
import torch.nn as nn
from collections import OrderedDict

from geovision.models import get_state_dict


class LayerNorm2d(nn.Module):
    """LayerNorm that supports 2D inputs (channels-first)"""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    """Global Response Normalization (GRN) for ConvNeXt V2"""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global feature aggregation
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block - modernized residual block"""
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = False,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) if use_grn else nn.Identity()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.grn(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt_Encoder(nn.Module):
    """ConvNeXt Encoder for image classification and dense prediction tasks

    Supports both ConvNeXt V1 and V2 architectures.
    ConvNeXt V2 adds Global Response Normalization (GRN) layers.
    """

    convnext_configs = {
        "tiny": {"dims": [96, 192, 384, 768], "depths": [3, 3, 9, 3]},
        "small": {"dims": [96, 192, 384, 768], "depths": [3, 3, 27, 3]},
        "base": {"dims": [128, 256, 512, 1024], "depths": [3, 3, 27, 3]},
        "large": {"dims": [192, 384, 768, 1536], "depths": [3, 3, 27, 3]},
        "xlarge": {"dims": [256, 512, 1024, 2048], "depths": [3, 3, 27, 3]},
    }

    def __init__(
        self,
        model_size: Literal["tiny", "small", "base", "large", "xlarge"] = "base",
        input_channels: int = 3,
        depths: Optional[Sequence[int]] = None,
        dims: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = False,  # Set to True for ConvNeXt V2
        weights_init: Literal["random", "torchvision", "torchgeo", "url", "path"] = "random",
        weights_param: Optional[str] = None,
    ):
        """
        Initialize ConvNeXt Encoder

        Parameters:
        -----------
        model_size : str
            One of ["tiny", "small", "base", "large", "xlarge"]
        input_channels : int
            Number of input image channels (default: 3)
        depths : Sequence[int], optional
            Number of blocks at each stage. Overrides model_size if provided.
        dims : Sequence[int], optional
            Feature dimensions at each stage. Overrides model_size if provided.
        drop_path_rate : float
            Stochastic depth rate (default: 0.0)
        layer_scale_init_value : float
            Layer scale initialization value (default: 1e-6)
        use_grn : bool
            Whether to use Global Response Normalization (ConvNeXt V2) (default: False)
        weights_init : str
            Weight initialization method
        weights_param : str, optional
            Parameter for weight initialization
        """
        super().__init__()

        # Use predefined config or custom depths/dims
        if depths is None or dims is None:
            assert model_size in self.convnext_configs, \
                f"model_size must be one of {list(self.convnext_configs.keys())}, got {model_size}"
            config = self.convnext_configs[model_size]
            depths = depths or config["depths"]
            dims = dims or config["dims"]

        assert len(depths) == len(dims), "depths and dims must have same length"

        self.depths = depths
        self.dims = dims
        self._num_input_channels = input_channels
        self._out_ch_per_layer = []
        self._downsampling_per_layer = []

        # Stem - aggressive downsampling like ResNet
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self._out_ch_per_layer.append(dims[0])
        self._downsampling_per_layer.append(4)

        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        cur = 0
        for i in range(len(depths)):
            # Downsampling layer between stages (except first)
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2),
                )
                setattr(self, f"downsample_{i}", downsample)
                self._downsampling_per_layer.append(2)
            else:
                self._downsampling_per_layer.append(1)

            # Build stage blocks
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                )
                for j in range(depths[i])
            ])
            setattr(self, f"stage_{i}", stage)
            self._out_ch_per_layer.append(dims[i])

            cur += depths[i]

        # Initialize weights
        self._init_weights()

        # Load pretrained weights if specified
        if weights_init != "random":
            state_dict = get_state_dict(weights_init, weights_param)
            self.load_state_dict(state_dict, strict=False)

    def _init_weights(self):
        """Initialize weights using truncated normal distribution"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning final stage features"""
        x = self.stem(x)

        for i in range(len(self.depths)):
            if i > 0:
                x = getattr(self, f"downsample_{i}")(x)
            x = getattr(self, f"stage_{i}")(x)

        return x

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning features from all stages (for dense prediction)"""
        features = []
        x = self.stem(x)
        features.append(x)

        for i in range(len(self.depths)):
            if i > 0:
                x = getattr(self, f"downsample_{i}")(x)
            x = getattr(self, f"stage_{i}")(x)
            features.append(x)

        return features


# Convenience constructors for ConvNeXt V1
def convnext_tiny(input_channels: int = 3, **kwargs):
    """ConvNeXt-T (Tiny) - 28M params"""
    return ConvNeXt_Encoder(model_size="tiny", input_channels=input_channels, use_grn=False, **kwargs)


def convnext_small(input_channels: int = 3, **kwargs):
    """ConvNeXt-S (Small) - 50M params"""
    return ConvNeXt_Encoder(model_size="small", input_channels=input_channels, use_grn=False, **kwargs)


def convnext_base(input_channels: int = 3, **kwargs):
    """ConvNeXt-B (Base) - 89M params"""
    return ConvNeXt_Encoder(model_size="base", input_channels=input_channels, use_grn=False, **kwargs)


def convnext_large(input_channels: int = 3, **kwargs):
    """ConvNeXt-L (Large) - 198M params"""
    return ConvNeXt_Encoder(model_size="large", input_channels=input_channels, use_grn=False, **kwargs)


def convnext_xlarge(input_channels: int = 3, **kwargs):
    """ConvNeXt-XL (X-Large) - 350M params"""
    return ConvNeXt_Encoder(model_size="xlarge", input_channels=input_channels, use_grn=False, **kwargs)


# Convenience constructors for ConvNeXt V2
def convnextv2_tiny(input_channels: int = 3, **kwargs):
    """ConvNeXt V2-T (Tiny) with GRN"""
    return ConvNeXt_Encoder(model_size="tiny", input_channels=input_channels, use_grn=True, **kwargs)


def convnextv2_small(input_channels: int = 3, **kwargs):
    """ConvNeXt V2-S (Small) with GRN"""
    return ConvNeXt_Encoder(model_size="small", input_channels=input_channels, use_grn=True, **kwargs)


def convnextv2_base(input_channels: int = 3, **kwargs):
    """ConvNeXt V2-B (Base) with GRN"""
    return ConvNeXt_Encoder(model_size="base", input_channels=input_channels, use_grn=True, **kwargs)


def convnextv2_large(input_channels: int = 3, **kwargs):
    """ConvNeXt V2-L (Large) with GRN"""
    return ConvNeXt_Encoder(model_size="large", input_channels=input_channels, use_grn=True, **kwargs)


def convnextv2_xlarge(input_channels: int = 3, **kwargs):
    """ConvNeXt V2-XL (X-Large) with GRN"""
    return ConvNeXt_Encoder(model_size="xlarge", input_channels=input_channels, use_grn=True, **kwargs)
