from typing import Literal, Optional, Sequence

import torch
import logging

from collections import OrderedDict
from geovision.models import get_state_dict
from geovision.models.blocks import ResidualBlock

logger = logging.getLogger(__name__)

class ResNetFeatureExtractor(torch.nn.Module):
    valid_weight_inits = ("random", "torchvision", "torchgeo", "url", "path")
    valid_residual_blocks = ("basic", "bottleneck", "basic_pre", "bottleneck_pre")
    valid_attention_blocks = ("none", "squeeze_and_excitation", "channel", "cbam")
    valid_identity_blocks = ("pointwise_strided", "pooled")
    resnet_layers = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3)
    }

    def __init__(
            self, 
            layers: Sequence[int] | Literal[18, 34, 50, 101, 152], 
            input_channels: int = 3,

            width_factor: Sequence[int] | int = 1,
            grouping_factor: Sequence[int] | int = 1,
            dilation_factor: Sequence[int] | int = 1,

            identity_block_name: Literal["pointwise_strided", "pooled"] = "pointwise_strided",
            residual_block_name: Literal["basic", "bottleneck", "basic_pre", "bottleneck_pre"] = "basic",
            attention_block_name: Literal["none", "squeeze_and_excitation", "channel", "cbam"] = "none", # For channel attention

            weights_init: Literal["random", "torchvision", "torchgeo", "url", "path"] = "random",
            weights_param: Optional[str] = None,
    ):
        """
        constructs an image encoder based on the ResNet family. parameter sequences can be specified to construct each layer, with the params applied\
        to all of the blocks within a layer, except for the :block and the :attn params, which are uniform throughout the network.
    
        the following args can be used to specify common architectures:
        - ResNet-50: (layers=50, block="bottleneck")
        - Wide_ResNet-50_2: (layers=50, block="bottleneck", width_factor=2)
        - SE-ResNeXt50_32x4d: (layers=50, block="bottleneck, grouping_factor=32, width_factor=2, attention_block_name="squeeze_and_excitation") 

        Parameters 
        - 
        :layers -> number of residual blocks per layer. alternatively specify a commonly used resnet variant.
        :width_factor -> multiplier for number of internal channels (width, mid_ch) per sub-layer. alternatively use a single int to apply the same \
                         width to all sub-layers. 
        :dilation_factor -> kernel dilation for Conv2d per layer. alternatively use a single int to apply the same dilation to all layers.
        :grouping_factor -> kernel grouping for Conv2d per layer. alternatively use a single int to apply the same grouping to all layers.

        :identity_block_name -> identity block, pointwise strided (conv1x1 with stride=2) or pooled (avg_pool with k=2 s=2)
        :residual_block_name -> residual block, basic or bottleneck. optionally postfixed by _pre to specify pre-act [norm->act->conv]
        :attention_block_name -> attention block to append to all residual blocks, defaults to "none".

        :weights_init -> source to initalize / load model weights.
        :weights_param -> param to specify weights .pt file to load, such as tv/torchgeo 'get_weight' Enum, pytorch-hub URL or a Path on the local fs.
        :input_channels -> number of channels in the input image, defaults to 3

        """
        super().__init__()

        assert isinstance(input_channels, int) and input_channels >= 1, f"config error, expected input channels to be int >= 1, got {input_channels}"
        self._num_input_channels = input_channels
        self._out_ch_per_layer = list() # To Construct Symmetric Decoders
        self._downsampling_per_layer = list() # To Construct Symmetric Decoders
        
        assert residual_block_name in self.valid_residual_blocks, \
            f"config error, expected :residual_block_name to be one of {self.valid_blocks}, got {residual_block_name}"
        if residual_block_name in ("basic", "basic_pre"):
            self._layer_constructor = self._basic_layer_constructor
        elif residual_block_name in ("bottleneck", "bottleneck_pre"):
            self._layer_constructor = self._bottleneck_layer_constructor
        
        if "pre" in residual_block_name:
            self._pre_act = True
        else:
            self._pre_act = False
        
        assert attention_block_name in self.valid_attention_blocks, \
            f"config error, expected :attention_block_name to be one of {self.valid_attention_blocks}, got {attention_block_name}"
        self._attn = attention_block_name

        assert identity_block_name in self.valid_identity_blocks, \
            f"config error, expected :identity_block_name to be one of {self.valid_identity_blocks}, got {identity_block_name}"
        self._idt = identity_block_name

        if isinstance(layers, int):
            assert layers in self.resnet_layers.keys(), f"config error, expected :layers to be one of {self.resnet_layers.keys()}, got {layers}"
            self._blocks_per_layer = self.resnet_layers[layers]
        else:
            assert isinstance(layers, Sequence), f"config error, expected :layers to be int or tuple[int,...], got {type(layers)}"
            self._blocks_per_layer = tuple(layers)
        
        if isinstance(dilation_factor, int):
            self._dilation_per_layer = [dilation_factor for _ in range(len(self._blocks_per_layer))]
        else:
            assert isinstance(dilation_factor, Sequence) and len(dilation_factor) == len(self._blocks_per_layer)
            self._dilation_per_layer = dilation_factor

        if isinstance(grouping_factor, int):
            self._grouping_per_layer = [grouping_factor for _ in range(len(self._blocks_per_layer))]
        else:
            assert isinstance(grouping_factor, Sequence) and len(grouping_factor) == len(self._blocks_per_layer)
            self._grouping_per_layer = grouping_factor

        if isinstance(width_factor, int):
            self._width_per_layer = [width_factor for _ in range(len(self._blocks_per_layer))]
        else:
            assert isinstance(width_factor, Sequence) and len(width_factor) == len(self._blocks_per_layer)
            self._width_per_layer = width_factor

        self.conv_1 = torch.nn.Sequential(OrderedDict({
            "conv_1" : torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            "norm_1" : torch.nn.BatchNorm2d(64),
            "act_1" : torch.nn.ReLU(inplace=True),
            "pool_1" : torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        }))
        self._out_ch_per_layer.append(64)

        if len(self._downsampling_per_layer) == 0:
            self._downsampling_per_layer.append(4)
        else:
            self._downsampling_per_layer[0] = 4

        self._layer_constructor()

        if weights_init != "random":
            state_dict = get_state_dict(weights_init, weights_param)
            if weights_init in ("torchvision", "torchgeo"):
                state_dict = rename_state_dict_keys(state_dict)
            self.load_state_dict(state_dict, strict = False)
        else:
            # TODO: Kaiming Init
            pass 

    def forward(self, x: torch.Tensor):
        for layer in self.children():
            x = layer(x)
        return x 
    
    def _basic_layer_constructor(self):
        def _basic_block_down(ch: int, layer_idx: int):
            return ResidualBlock(
                in_ch=ch, 
                mid_ch=2*ch*self._width_per_layer[layer_idx], 
                out_ch=2*ch, 
                groups=self._grouping_per_layer[layer_idx], 
                dilation=self._dilation_per_layer[layer_idx], 
                residual_path=residual_block, 
                identity_path=self._idt, 
                attention_path=self._attn
            )
        
        def _basic_block_same(ch: int, layer_idx: int):
            return ResidualBlock(
                in_ch=ch, 
                mid_ch=ch*self._width_per_layer[layer_idx], 
                out_ch=ch, 
                groups=self._grouping_per_layer[layer_idx], 
                dilation=self._dilation_per_layer[layer_idx], 
                residual_path=residual_block, 
                identity_path="identity", 
                attention_path=self._attn
            )

        residual_block = "basic_pre" if self._pre_act else "basic"
        num_channels = 64
        for layer_id, num_blocks in enumerate(self._blocks_per_layer):
            layer = torch.nn.Sequential()

            if layer_id == 0:
                layer.add_module("1", _basic_block_same(num_channels, layer_id))
                self._downsampling_per_layer.append(1)
            else:
                layer.add_module("1", _basic_block_down(num_channels, layer_id))
                self._downsampling_per_layer.append(2)
                num_channels*=2

            self._out_ch_per_layer.append(num_channels)
            
            for block_id in range(2, num_blocks+1):
                layer.add_module(str(block_id), _basic_block_same(num_channels, layer_id))

            setattr(self, f"conv_{layer_id+2}", layer)

    def _bottleneck_layer_constructor(self):
        def _bottleneck_block_down(in_ch: int, mid_ch: int, out_ch: int, layer_idx: int):
            return ResidualBlock(
                in_ch = in_ch, 
                mid_ch = mid_ch*self._width_per_layer[layer_idx],
                out_ch = out_ch, 
                groups = self._grouping_per_layer[layer_idx],
                dilation = self._dilation_per_layer[layer_idx],
                residual_path = residual_block,
                identity_path = self._idt,
                attention_path = self._attn
            )

        def _bottleneck_block_same(in_ch: int, mid_ch: int, out_ch: int, layer_idx: int):
            return ResidualBlock(
                in_ch = in_ch, 
                mid_ch = mid_ch*self._width_per_layer[layer_idx],
                out_ch = out_ch, 
                groups = self._grouping_per_layer[layer_idx],
                dilation = self._dilation_per_layer[layer_idx],
                residual_path = residual_block,
                identity_path = "identity",
                attention_path = self._attn
            )
        
        in_channels, mid_channels, out_channels = 64, 64, 256
        residual_block = "bottleneck_pre" if self._pre_act else "bottleneck"

        for layer_id, num_blocks in enumerate(self._blocks_per_layer):
            layer = torch.nn.Sequential()
            if layer_id == 0:
                layer.add_module("1", _bottleneck_block_same(in_channels, mid_channels, out_channels, layer_id))
                self._downsampling_per_layer.append(1)
            else:
                layer.add_module("1", _bottleneck_block_down(in_channels, mid_channels, out_channels, layer_id))
                self._downsampling_per_layer.append(2)

            in_channels = out_channels
            self._out_ch_per_layer.append(out_channels)

            for block_id in range(2, num_blocks+1):
                layer.add_module(str(block_id), _bottleneck_block_same(in_channels, mid_channels, out_channels, layer_id))
            mid_channels *= 2
            out_channels *= 2

            setattr(self, f"conv_{layer_id+2}", layer) 
    
def rename_state_dict_keys(state_dict: dict) -> dict:
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
    return new_state_dict