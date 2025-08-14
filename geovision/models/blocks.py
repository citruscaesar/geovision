from typing import Any, Optional, Literal, Callable

import torch
from collections import OrderedDict
from geovision.models import get_state_dict

class LinearDecoderBlock(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int, weights_init: Literal["random", "torchvision", "torchgeo", "url", "path"], weights_param: Optional[str] = None):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = torch.nn.Linear(in_ch, out_ch, bias=True)

        if weights_init != "random":
            self.load_state_dict(get_state_dict(weights_init, weights_param), strict = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.avgpool(x).flatten(1))

class ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            mid_ch: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            residual_path: Literal["basic", "bottleneck", "baisc_pre", "bottleneck_pre"] = "basic",
            identity_path: Literal["identity", "pointwise_strided", "pooled"] = "identity",
            attention_path: Literal["none", "squeeze_and_excitation", "channel", "cbam"] = "none"
        ):
        super().__init__()
        assert residual_path in ("basic", "bottleneck", "baisc_pre", "bottleneck_pre")
        assert identity_path in ("identity", "pointwise_strided", "pooled")
        assert attention_path in ("none", "squeeze_and_excitation", "channel", "cbam")

        if identity_path == "identity":
            if mid_ch == out_ch:
                self.identity = torch.nn.Identity()
            else:
                self.identity = torch.nn.Sequential(OrderedDict({
                    "conv_1": torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    "norm_1": torch.nn.BatchNorm2d(out_ch)
                }))

        elif identity_path == "pointwise_strided":
            self.identity = torch.nn.Sequential(OrderedDict({
                "conv_1": torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False),
                "norm_1": torch.nn.BatchNorm2d(out_ch)
            }))

        elif identity_path == "pooled":
            self.identity = torch.nn.Sequential(OrderedDict({
                "pool_1": torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                "conv_1": torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                "norm_1": torch.nn.BatchNorm2d(out_ch)
            }))

        stride = 2 if identity_path != "identity" else 1

        if residual_path == "basic":
            mid_ch = mid_ch or in_ch
            self.apply_act = True
            self.residual = torch.nn.Sequential(OrderedDict({
                "conv_1": torch.nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
                "norm_1": torch.nn.BatchNorm2d(mid_ch),
                "act_1": torch.nn.ReLU(inplace = True),
                "conv_2": torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
                "norm_2": torch.nn.BatchNorm2d(out_ch),
            }))

        elif residual_path == "basic_pre":
            mid_ch = mid_ch or in_ch
            self.apply_act = False
            self.residual = torch.nn.Sequential(OrderedDict({
                "norm_1": torch.nn.BatchNorm2d(in_ch),
                "act_1": torch.nn.ReLU(inplace = True),
                "conv_1": torch.nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
                "norm_2": torch.nn.BatchNorm2d(mid_ch),
                "act_2": torch.nn.ReLU(inplace = True),
                "conv_2":  torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
            }))

        elif residual_path == "bottleneck":
            mid_ch = mid_ch or in_ch // 4
            self.apply_act = True
            self.residual = torch.nn.Sequential(OrderedDict({
                "conv_1": torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
                "norm_1": torch.nn.BatchNorm2d(mid_ch),
                "act_1": torch.nn.ReLU(inplace = True),
                "conv_2": torch.nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False),
                "norm_2": torch.nn.BatchNorm2d(mid_ch),
                "act_2": torch.nn.ReLU(inplace = True),
                "conv_3": torch.nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                "norm_3": torch.nn.BatchNorm2d(out_ch),
            }))

        elif residual_path == "bottleneck_pre":
            mid_ch = mid_ch or in_ch // 4
            self.apply_act = False
            self.residual =  torch.nn.Sequential(OrderedDict({
                "norm_1": torch.nn.BatchNorm2d(in_ch),
                "act_1": torch.nn.ReLU(inplace = True),
                "conv_1": torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
                "norm_2": torch.nn.BatchNorm2d(mid_ch),
                "act_2": torch.nn.ReLU(inplace = True),
                "conv_2": torch.nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False),
                "norm_3": torch.nn.BatchNorm2d(mid_ch),
                "act_3": torch.nn.ReLU(inplace = True),
                "conv_3": torch.nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            }))
        
        if attention_path == "squeeze_and_excitation":
            self.attention = SqueezeAndExcitationBlock(in_ch, mid_ch=in_ch*4)
        else:
            self.attention = torch.nn.Identity()

    def forward(self, x):
        out = self.identity(x) + self.attention(self.residual(x))
        if self.apply_act:
            return torch.relu(out)
        return out

class MBConvBlock(torch.nn.Module): ...

class ConvNeXtBlock(torch.nn.Module):
    def __init__(self, in_ch: int, mid_ch: int):
        super().__init__()
        self.residual = torch.nn.Sequential(OrderedDict([
            ("conv_1", torch.nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding = 'same')),
            ("norm_1", torch.nn.LayerNorm(in_ch)),
            ("conv_2", torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0)),
            ("act_1", torch.nn.GELU()),
            ("conv3", torch.nn.Conv2d(mid_ch, in_ch, kernel_size=1, stride=1, padding=0))
        ])) 

class SqueezeAndExcitationBlock(torch.nn.Module):
    def __init__(self, in_ch: int, mid_ch: int) -> None:
        super().__init__()
        self.channel_weights = torch.nn.Sequential(OrderedDict({
            "pool" : torch.nn.AdaptiveAvgPool2d((1,1)),
            "fc_1" : torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, groups=1, bias = True),
            "act_1" : torch.nn.ReLU(inplace=True),
            "fc_2" : torch.nn.Conv2d(mid_ch, in_ch, kernel_size=1, stride=1, padding=0, groups=1, bias = True),
            "act_2" : torch.nn.ReLU(inplace=True),
        }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_weights(x)

class ChannelAttentionBlock(torch.nn.Module): ...
class SpatialAttentionBlock(torch.nn.Module): ...
class CBAMBlock(torch.nn.Module): ...
class GlobalContextBlock(torch.nn.Module): ...
class CoordinateAttentionBlock(torch.nn.Module): ...