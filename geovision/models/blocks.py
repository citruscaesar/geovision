from typing import Optional, Literal

import torch
from collections import OrderedDict

class ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            block: Literal["basic", "bottleneck"],
            in_ch: int,
            mid_ch: int,
            out_ch: int,
            groups: int = 1,
            downsample: bool = False,
            identity_path: Literal["conv1x1_norm", "pool2x2_conv1x1"] = "conv1x1_norm",
            block_sequence: Literal["conv_norm_act", "norm_act_conv"] = "conv_norm_act",
            #channel_attention: Optional[torch.nn.Module] = None

            # downsample: Literal["strided", "dilated"]
            # downsampling: int
            # for strided, downsampling can be 1 or 2, since kernel size is 3
            # for dilated, downsampling should be provided =  image_height // 4 for :2x downsampling, or image_height // 2 for 4x downsampling
            #             -> dilated downsampling on the identity path needs a 2x2 or 3x3 kernel 
        ):
        super().__init__()
        assert block in ("basic", "bottleneck")
        assert block_sequence in ("conv_norm_act", "norm_act_conv")
        self.block_sequence = block_sequence

        if downsample:
            assert identity_path in ("conv1x1_norm", "pool2x2_conv1x1"), \
                ":identity_path must be one of conv1x1_norm, pool2x2_conv1x1 when :downsample is True"
            self.identity = torch.nn.Sequential()
            if identity_path == "conv1x1_norm":
                self.identity.add_module("conv_1", torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias = False))
                self.identity.add_module("norm_1", torch.nn.BatchNorm2d(out_ch))
            elif identity_path == "pool2x2_conv1x1":
                self.identity.add_module("pool_1", torch.nn.AvgPool2d(kernel_size=2, stride=2))
                self.identity.add_module("conv_1", torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias = False))
        elif in_ch != out_ch:
            assert identity_path == "conv1x1_norm", "identity path must be conv_norm when :in_ch!=:out_ch without downsampling"
            self.identity = torch.nn.Sequential()
            self.identity.add_module("conv_1", torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias = False))
            self.identity.add_module("norm_1", torch.nn.BatchNorm2d(out_ch))
        else:
            self.identity = torch.nn.Identity()
        
        if block == "basic":
            if block_sequence == "conv_norm_act":
                self.residual = torch.nn.Sequential(OrderedDict({
                    "conv_1" : torch.nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
                    "norm_1" : torch.nn.BatchNorm2d(mid_ch),
                    "act_1" : torch.nn.ReLU(inplace = True),
                    "conv_2" :  torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                    "norm_2" : torch.nn.BatchNorm2d(out_ch),
                }))
            elif block_sequence == "norm_act_conv":
                self.residual = torch.nn.Sequential(OrderedDict({
                    "norm_1" : torch.nn.BatchNorm2d(in_ch),
                    "act_1" : torch.nn.ReLU(inplace = True),
                    "conv_1" : torch.nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
                    "norm_2" : torch.nn.BatchNorm2d(mid_ch),
                    "act_2" : torch.nn.ReLU(inplace = True),
                    "conv_2" :  torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
                }))
        else:
            if block_sequence == "conv_norm_act":
                self.residual = torch.nn.Sequential(OrderedDict({
                    "conv_1": torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    "norm_1": torch.nn.BatchNorm2d(mid_ch),
                    "act_1": torch.nn.ReLU(inplace = True),
                    "conv_2": torch.nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=groups, bias=False),
                    "norm_2": torch.nn.BatchNorm2d(mid_ch),
                    "act_2": torch.nn.ReLU(inplace = True),
                    "conv_3": torch.nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    "norm_3": torch.nn.BatchNorm2d(out_ch),
                }))
            elif block_sequence == "norm_act_conv":
                self.residual = torch.nn.Sequential(OrderedDict({
                    "norm_1": torch.nn.BatchNorm2d(in_ch),
                    "act_1": torch.nn.ReLU(inplace = True),
                    "conv_1": torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    "norm_2": torch.nn.BatchNorm2d(mid_ch),
                    "act_2": torch.nn.ReLU(inplace = True),
                    "conv_2": torch.nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2 if downsample else 1, padding=1, groups=groups, bias=False),
                    "norm_3": torch.nn.BatchNorm2d(mid_ch),
                    "act_3": torch.nn.ReLU(inplace = True),
                    "conv_3": torch.nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                }))
        
        #if channel_attention is not None:
            #self.attn = channel_attention 
        #else:
            #self.attn = torch.nn.Identity()
        
    def forward(self, x):
        out = self.identity(x) + self.residual(x)
        if self.block_sequence == "conv_norm_act":
            return torch.relu(out)
        return out


class MBConvBlock(torch.nn.Module):
    pass

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

class ChannelAttentionBlock(torch.nn.Module):
    pass

class SqueezeAndExcitationBlock(torch.nn.Module):
    def __init__(self, in_ch: int, mid_ch: int) -> None:
        super().__init__()
        self.attn = torch.nn.Sequential(OrderedDict([
            ("pool", torch.nn.AdaptiveAvgPool2d((1,1))),
            ("fc_1", torch.nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, groups=1, bias = True)),
            ("act_1", torch.nn.ReLU()),
            ("fc_2", torch.nn.Conv2d(mid_ch, in_ch, kernel_size=1, stride=1, padding=0, groups=1, bias = True)),
            ("act_2", torch.nn.ReLU()),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_attention(x)