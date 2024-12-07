from typing import Literal, Optional, Iterable
from collections.abc import Callable

import torch
import torchvision
from pathlib import Path
from .blocks import ResidualBlock

import logging
logger = logging.getLogger(__name__)

class ResNetFeatureExtractor(torch.nn.Module):
    valid_weights = ("kaiming", "torchvision", "torchgeo", "url", "path")
    def __init__(
            self, 
            layers: Literal[18, 34, 50, 101, 152] | Iterable[int], 
            block: Optional[Literal["basic", "bottleneck"]] = None,
            weights: Literal["kaiming", "torchvision", "torchgeo", "url", "path"] = "kaiming",
            weights_param: Optional[str] = None,
            input_channels: int = 3,
            channel_grouping: int = 1,
            channel_multiplier: int = 1,
            # channel_attention: Optional[Literal["se", "ca", "eca", "cbam"]] = None 
            # kernel_dilation: bool = False, -> dilation for downsampling = input / 4
            # stem_kernel_size: Literal = [7,5,3]
            # stem_pooling: Optional[Literal["max", "avg", "none"]] 
    ):
        super().__init__()

        if isinstance(layers, Iterable):
            assert block is not None and block in ("basic", "bottleneck"), \
                f"config error, expected :block to be basic or bottleneck when :(iterable)layers is provided, got ({type(block)}){block}"
            #assert len(layers) == 4, f"config error, expected :(iterable)layers to be of len 4, got len {(len(layers))}"
            self.layers = layers
            self.layer_constructor = self._basic_layer_constructor if block == "basic" else self._bottleneck_layer_constructor
        elif isinstance(layers, int):
            assert layers in (18, 34, 50, 101, 152), f"not implemented error, expected :layers to be one of 18, 34, 50, 101, 152, got {layers}"
            if layers in (18, 34):
                self.layer_constructor = self._basic_layer_constructor
            else:
                self.layer_constructor = self._bottleneck_layer_constructor
            if layers == 18:
                self.layers = (2, 2, 2, 2)
            elif layers == 34:
                self.layers = (3, 4, 6, 3)
            elif layers == 50:
                self.layers = (3, 4, 6, 3)
            elif layers == 101:
                self.layers = (3, 4, 23, 3)
            elif layers == 152:
                self.layers = (3, 8, 36, 3)
        else:
            raise AssertionError(f"config error, expected :layers to be int or tuple[int,...], got {type(layers)}")

        assert weights in self.valid_weights, f"config error, expected :weights to be one of {self.valid_weights}, got {weights}"
        if weights == "kaiming":
            assert weights_param is None, \
                f"config error, expected :weights_param to be None when :weights are kaiming (init), got {type(weights_param)}"
            self.weights = None 
        else:
            assert weights_param is not None, \
                f"config error, expected :weights_param to not None when :weights are {weights}, got {type(weights_param)}"
            if weights == "torchvision":
                self.weights = torchvision.models.get_weight(weights_param)
            elif weights == "torchgeo":
                raise NotImplementedError("loading torchgeo weights is not implemented yet")
            elif weights == "url":
                assert isinstance(weights_param, str), f"config error, expected :weights_param to be a valid url when :weights is url, got {weights_param}"
                self.weights = weights_param
            elif weights == "path":
                weights_param: Path = Path(weights_param).expanduser().resolve()
                assert weights_param.is_file(), f"invalid path error, :weights_param [{weights_param}] does not point to a valid file on this fs"
                self.weights = weights_param
        
        assert isinstance(channel_multiplier, int) and channel_multiplier >= 1
        assert isinstance(channel_grouping, int) and channel_grouping >= 1

        self._output_channels = list()

        self.conv_1 = torch.nn.Sequential()
        self.conv_1.add_module("conv_1", torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.conv_1.add_module("norm_1", torch.nn.BatchNorm2d(64))
        self.conv_1.add_module("act_1", torch.nn.ReLU(inplace=True))
        self.conv_1.add_module("pool_1", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self._output_channels.append(64)

        self.layer_constructor(self.layers, channel_grouping, channel_multiplier)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if self.weights is not None:
            if isinstance(self.weights, torchvision.models.WeightsEnum):
                self.load_state_dict(self._rename_torch_weights(self.weights.get_state_dict()), strict = False)
            elif isinstance(self.weights, Path):
                self.load_state_dict(torch.load(self.weights, weights_only=True), strict = False)
            else:
                self.load_state_dict(self._rename_torch_weights(torch.hub.load_state_dict_from_url(self.weights, weights_only=True), strict=False))
        else:
            # TODO: write kaiming init
            pass

    def forward(self, x: torch.Tensor):
        for layer in self.children():
            x = layer(x)
        return x 
    
    @property
    def output_channels(self) -> tuple[int,...]:
        return tuple(self._output_channels)

    def _basic_layer_constructor(self, blocks: tuple[int, ...], groups: int, multiplier: int):
        def _basic_block(channels: int, downsampling: bool, dilation_factor: int = 1):
            if downsampling:
                return ResidualBlock("basic", channels, 2*channels*multiplier, 2*channels, groups, True, "conv1x1_norm", "conv_norm_act")
            else:
                return ResidualBlock("basic", channels, channels*multiplier, channels, groups, False, "conv1x1_norm", "conv_norm_act")

        num_channels = 64
        for layer_id, num_blocks in enumerate(blocks):
            setattr(self, f"conv_{layer_id+2}", torch.nn.Sequential())
            layer = getattr(self, f"conv_{layer_id+2}")

            if layer_id != 0:
                layer.add_module("1", _basic_block(channels=num_channels, downsampling=True, dilation_factor = 64 // 4*(layer_id+1)))
                num_channels*=2
            else:
                layer.add_module("1", _basic_block(channels=num_channels, downsampling=False))

            for block_id in range(2, num_blocks+1):
                layer.add_module(f"{block_id}", _basic_block(channels=num_channels, downsampling=False))
            self._output_channels.append(num_channels)

    def _bottleneck_layer_constructor(self, blocks: tuple[int, ...], groups:int, multiplier:int):
        def _bottleneck_block(in_ch: int, mid_ch: int, out_ch: int, downsampling: bool, dilation_factor: int  = 1):
            if downsampling:
                return ResidualBlock("bottleneck", in_ch, mid_ch*multiplier, out_ch, groups, True, "conv1x1_norm", "conv_norm_act")
            else:
                return ResidualBlock("bottleneck", in_ch, mid_ch*multiplier, out_ch, groups, False, "conv1x1_norm", "conv_norm_act")

        in_channels = 64
        mid_channels = 64 if groups == 1 else 128
        out_channels = 256
        for layer_id, num_blocks in enumerate(blocks):
            setattr(self, f"conv_{layer_id+2}", torch.nn.Sequential()) 
            layer = getattr(self, f"conv_{layer_id+2}")

            layer.add_module("1", _bottleneck_block(in_channels, mid_channels, out_channels, downsampling=(layer_id!=0), dilation_factor=64 // 4*(layer_id+1)))
            in_channels = out_channels
            for block_id in range(2, num_blocks+1):
                layer.add_module(f"{block_id}", _bottleneck_block(in_channels, mid_channels, out_channels, downsampling=False))
            self._output_channels.append(out_channels)
            mid_channels *= 2
            out_channels *= 2

                           
    @staticmethod
    def _rename_torch_weights(state_dict: dict) -> dict:
        def numeric_char_at(string: str) -> int:
            for i, c in enumerate(string):
                if c.isnumeric():
                    return i
            return len(string) 

        def add_underscore_bw_char_and_int(string: str) -> str:
            num_idx = numeric_char_at(string)
            if num_idx != len(string):
                return f"{rename_layer(string[:num_idx])}_{string[num_idx:]}"
            else:
                return string
        
        def rename_layer(string: str) -> str:
            if string == "layer":
                return "conv"
            elif string == "bn":
                return "norm"
            return string

        new_state_dict = dict()
        for name, value in state_dict.items():
            parts = name.split('.') 

            # conv1.weight -> conv_1.weight
            if len(parts) == 2: 
                if 'fc' in parts[0]:
                    new_state_dict[name] = value
                else:
                    new_state_dict[f"conv_1.{add_underscore_bw_char_and_int(parts[0])}.{parts[1]}"] = value

            # layer1.2.bn2.bias -> conv_2_3.residual.bn_2.bias
            else: 
                first, second, third = parts[0], parts[1], parts[2]

                # layer1 -> conv_2
                num_idx = numeric_char_at(first)
                if num_idx != len(first):
                    first = f"{rename_layer(first[:num_idx])}_{int(first[num_idx:])+1}"

                # 2 -> 3
                second = int(second) + 1

                # downsample.0.weights -> identity.conv_1.weights
                if third == "downsample":
                    rename_fourth = {"0": "conv_1", "1": "norm_1"}
                    fourth = rename_fourth[parts[3]]
                    key = '.'.join([f"{first}.{second}", "identity", f"{fourth}"] + list(parts[4:]))
                
                # bn2.bias -> residual.bn_2.bias
                else:
                    third = add_underscore_bw_char_and_int(third)
                    key = '.'.join([f"{first}.{second}", "residual", f"{third}"] + list(parts[3:]))
                new_state_dict[key] = value

                # NOTE: batchnorm being placed beforce conv leads to shape mismatch;
                # TODO: keep the first however many required parameters, discard the rest
                # if "_1.residual.bn_" in key and "conv_2_1" not in key:
                    # try:
                        # value = value[:len(value)//2]
                    # except Exception as err:
                        # print(name, value, err)
                        # continue
                # TODO: or perhaps remove all batchnorm parameters -> relearn with very small lr
        return new_state_dict