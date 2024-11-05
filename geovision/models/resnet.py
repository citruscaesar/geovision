from typing import Literal, Optional

import torch
import torchvision
from .blocks import (
    BasicResidualBlock,
    BottleneckResidualBlock
)

class ResNet(torch.nn.Module):
    def __init__(
            self, 
            version: Optional[Literal[18, 34, 50, 101, 152]] = 18, 
            weights: Literal["random", "torch"] = "random", 
            config: Optional[tuple[int, ...]] = None,
            num_classes: int = 1000,
        ):
        super().__init__()
        assert version in (18, 34, 50, 101, 152, None), f"not implemented error, expected version to be one of 18, 34, 50, 101, 152 or None, got {version}"
        assert (isinstance(config, tuple) and len(config) == 4) or config is None, f"value error, expected :config to be a None or a tuple of length 4, got {config}"
        assert (version and weights) or config, "misconfiguration error, expected one of :version + :weights or :config to be provided, both are None"
        if version is not None:
            assert weights in ("random", "torch"), f"not implemented error, expected :weights to be one of random or torch, got {weights}"

        self.feature_extractor = torch.nn.Sequential()
        self.feature_extractor.add_module("conv_1", torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False))
        self.feature_extractor.add_module("bn_1", torch.nn.BatchNorm2d(64))
        self.feature_extractor.add_module("relu_1", torch.nn.ReLU())
        self.feature_extractor.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        assert version == 18, f"not implemented error, expected :version to be 18, got {version}"
        self.feature_extractor.add_module("conv_2_1", BasicResidualBlock(64))
        self.feature_extractor.add_module("conv_2_2", BasicResidualBlock(64))
        self.feature_extractor.add_module("conv_3_1", BasicResidualBlock(64, 128, 128, downsample=True))
        self.feature_extractor.add_module("conv_3_2", BasicResidualBlock(128, 128, 128))
        self.feature_extractor.add_module("conv_4_1", BasicResidualBlock(128, 256, 256, downsample=True))
        self.feature_extractor.add_module("conv_4_2", BasicResidualBlock(256, 256, 256))
        self.feature_extractor.add_module("conv_5_1", BasicResidualBlock(256, 512, 512, downsample=True))
        self.feature_extractor.add_module("conv_5_2", BasicResidualBlock(512, 512, 512))
        self.feature_extractor.add_module("avgpool", torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.classifier = torch.nn.Linear(512, num_classes)

        if weights == "random":
            pass
        elif weights == "torch":
            state_dict = self.get_resnet_state_dict_from_torch(torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict())
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError

        # elif version == 34:
            # block = BasicResidualBlock
            # config = (3, 4, 6, 3)
        # elif version == 50: 
            # block = BottleneckResidualBlock

        # for layer_idx, num_blocks in enumerate(config):
            # if layer_idx == 0:
                # in_channels, mid_channels, out_channels = 64, 64, 64
            # else:
                # in_channels, out_channels = 64 * layer_idx
                # out_channels = 64 * (layer_idx + 1)

            # downsample = True if layer_idx > 0 else False
            # for block_idx in range(num_blocks):
                # self.feature_extractor.add_module(f"conv_{layer_idx+2}_{block_idx+1}", block(in_channels, mid_channels, out_channels, downsample))
                # downsample = False
                
    def forward(self, x: torch.Tensor):
       return self.classifier(torch.flatten(self.feature_extractor(x), start_dim=1))

    def get_resnet_state_dict_from_torch(self, state_dict: dict) -> dict:
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
            return "conv" if string == "layer" else string

        new_state_dict = dict()
        for name, value in state_dict.items():
            parts = name.split('.')
            if len(parts) == 2:
                # e.g. conv1.weight -> conv_1.weight or fc.bias -> fc.bias
                parts[0] = add_underscore_bw_char_and_int(parts[0])
                new_state_dict[f"feature_extractor.{parts[0]}.{parts[1]}"] = value
            else:
                # e.g. layer1.2.bn2.bias -> layer_2_3.residual.bn_2.bias
                first, second, third = parts[0], parts[1], parts[2]
                num_idx = numeric_char_at(first)
                if num_idx != len(first):
                    first = f"{rename_layer(first[:num_idx])}_{int(first[num_idx:])+1}"
                second = int(second) + 1
                if third == "downsample":
                    rename_fourth = {"0": "conv_1", "1": "bn_1"}
                    fourth = rename_fourth[parts[3]]
                    key = '.'.join(["feature_extractor", f"{first}_{second}", "identity", f"{fourth}"] + list(parts[4:]))
                else:
                    third = add_underscore_bw_char_and_int(third)
                    key = '.'.join(["feature_extractor", f"{first}_{second}", "residual", f"{third}"] + list(parts[3:]))

                # To handle batchnorm being placed beforce conv, leading to shape mismatch, only use the first half of means and vars
                if "_1.residual.bn_1" in key and "conv_2_1" not in key:
                    value = value[:len(value)//2]

                new_state_dict[key] = value
        return new_state_dict