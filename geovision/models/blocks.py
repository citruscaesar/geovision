from typing import Optional

import torch
from collections import OrderedDict

class BasicResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, mid_channels: Optional[int] = None, out_channels: Optional[int] = None, downsample: bool = False, groups: int = 1):
        super().__init__()
        assert isinstance(in_channels, int), f"type error, expected :in_channels to be of type int, got {type(in_channels)}"
        assert isinstance(mid_channels, None | int), f"type error, expected :mid_channels to be of type None or int, got {type(mid_channels)}"
        assert isinstance(out_channels, None | int), f"type error, expected :out_channels to be of type None or int, got {type(out_channels)}"
        assert isinstance(downsample, bool), f"type error, expected :downsample to be of type bool, got {type(downsample)}"

        if mid_channels is None:
            mid_channels = in_channels 

        if out_channels is None:
            out_channels = in_channels

        if not downsample and in_channels == out_channels:
            self.identity = torch.nn.Identity()
        else:
            self.identity = torch.nn.Sequential()
            if downsample:
                self.identity.add_module("avg_pool", torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            if in_channels != out_channels:
                self.identity.add_module("conv_1", torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.residual = torch.nn.Sequential(OrderedDict({
            "bn_1" : torch.nn.BatchNorm2d(num_features=in_channels),
            "relu_1" : torch.nn.ReLU(),
            "conv_1" : torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            "bn_2" : torch.nn.BatchNorm2d(num_features=mid_channels),
            "relu_2" : torch.nn.ReLU(),
            "conv_2" :  torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x) + self.residual(x)
    
class BottleneckResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, mid_channels: Optional[int] = None, out_channels: Optional[int] = None, downsample: bool = False, groups: int = 1):
        super().__init__()
        assert isinstance(in_channels, int), f"type error, expected :in_channels to be of type int, got {type(in_channels)}"
        assert isinstance(mid_channels, None | int), f"type error, expected :mid_channels to be of type None or int, got {type(mid_channels)}"
        assert isinstance(out_channels, None | int), f"type error, expected :out_channels to be of type None or int, got {type(out_channels)}"
        assert isinstance(downsample, bool), f"type error, expected :downsample to be of type bool, got {type(downsample)}"

        if mid_channels is None:
            assert in_channels % 4 == 0, f"value error, :mid_channels not specified, default to :in_channels/4, thus expected :in_channels to be divisible by 4, got {in_channels}"
            mid_channels = in_channels // 4

        if out_channels is None:
            out_channels = in_channels

        if not downsample and in_channels == out_channels:
            self.identity = torch.nn.Identity()
        else:
            self.identity = torch.nn.Sequential()
            if downsample:
                self.identity.add_module("avg_pool", torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            if in_channels != out_channels:
                self.identity.add_module("conv", torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
 
        self.residual = torch.nn.Sequential(OrderedDict({
            "bn_1": torch.nn.BatchNorm2d(num_features=in_channels),
            "relu_1": torch.nn.ReLU(),
            "conv_1": torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            "bn_2": torch.nn.BatchNorm2d(num_features=mid_channels),
            "relu_2": torch.nn.ReLU(),
            "conv_2": torch.nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=groups, bias=False),
            "bn_3": torch.nn.BatchNorm2d(num_features=mid_channels),
            "relu_3": torch.nn.ReLU(),
            "conv_3": torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        }))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x) + self.residual(x)