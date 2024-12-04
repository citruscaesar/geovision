from typing import Literal
from collections import OrderedDict
import torch

def __init__(self, in_ch: int, out_ch: int, upsampling: Literal["transposed", "nearest", "bilinear", "bicubic"]):
        super().__init__()
        # :in_ch are num_channels from previous layer
        # encoder output will already have out_ch
        self.skip = torch.nn.Conv1d(out_ch, out_ch//2, kernel_size=1, stride=1, padding=0, bias=False) 

        self.upsample = torch.nn.Sequential() 
        if upsampling == "transposed":
            self.upsample.add_module("tconv_1", torch.nn.ConvTranspose2d(in_ch, out_ch//2, kernel_size=2, stride=2, padding=0))
        else:
            self.upsample.add_module("up_1", torch.nn.Upsample(scale_factor=2, mode = "upsampling"))
            self.upsample.add_module("conv_1", torch.nn.Conv2d(in_ch, out_ch//2, kernel_size=3, stride=1, padding=1))
        # else:
            # self.upsample.add_module("up_1", torch.nn.)
    
    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        out = torch.concat([self.skip(skip_x), self.upsample(x)], dim = 1)
        # spatially upsample x to 2x 
        # spectrally downsample x to x/4 
        # spectrally downsample skip_x to skip_x/2
        # spectrally concat x/4 and skip_x/2 and project to x 

class CentralUnetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

class UNetDecoder(torch.nn.Module):
    def __init__(
            self, 
            out_channels: int,
            channels_per_layer: list[int], 
    ):
        super().__init__()

        assert isinstance(channels_per_layer, list)
        assert len(channels_per_layer) == 5
        conv_5_channels, conv_4_channels, conv_3_channels, conv_2_channels, conv_1_channels = channels_per_layer
    
        central_layers = channels_per_layer[4] // 2

        self.central_block = torch.nn.Sequential(OrderedDict([
            ("pool_1", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # spatial down
            ("conv_1", torch.nn.Conv2d(conv_5_channels, conv_5_channels*2, 1, 1, 0, bias = False)), # spectral up 
            ("conv_2", torch.nn.Conv2d(conv_5_channels*2, conv_5_channels, 1, 1)) # spectral down
            ("up_1", torch.nn.Upsample(scale_factor=2, mode = "nearest")) # spatial up
        ]))

    def forward(self, x_1, x_2, x_3, x_4, x_5):
        d_4 = self.central_block(x_5)
        
