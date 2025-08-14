from typing import Any, Sequence, Callable, Literal

import torch
from collections import OrderedDict

class UNetDecoderBlock(torch.nn.Module):
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int, 
            upsampling: str,
            upsampling_factor: int,
            conv_block: Callable[..., torch.nn.Module],
            conv_block_params: dict[str, Any]
        ):
        super().__init__()
        assert upsampling in ("transposed", "bilinear", "nearest", "bicubic")
        assert isinstance(upsampling_factor, int) and upsampling_factor >= 1

        self.reproj =  torch.nn.Conv2d(in_ch*2, out_ch, kernel_size = 1, stride = 1, padding = 0, bias = False)
        if upsampling_factor == 1:
            self.upsample = torch.nn.Identity()
        else:
            if upsampling == "transposed":
                self.upsample = torch.nn.Sequential(OrderedDict({
                    "tconv" : torch.nn.ConvTranspose2d(out_ch, out_ch, upsampling_factor, upsampling_factor, bias = False),
                    "act" : torch.nn.ReLU(inplace=True),
                    "norm" : torch.nn.BatchNorm2d(out_ch),
                }))
            else:
                self.upsample = torch.nn.Upsample(scale_factor=upsampling_factor, mode=upsampling)
        self.conv = conv_block(in_ch = out_ch, out_ch = out_ch, **conv_block_params)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        # assert x.shape == skip_x.shape
        # N x [in_ch, in_ch] x H x W -> N x out_ch x 2H x 2W
        x = torch.cat((x, skip_x), dim = 1)
        x = self.reproj(x)
        x = self.upsample(x)
        x = self.conv(x)
        return x

class UNetDecoder(torch.nn.Module):
    # NOTE: UNet: encoder + central_block + decoder [encoder and decoder are symmetric in tensor dimensions]
    # NOTE: encoder: (in_ch x H x W) -> (C' x H' x W') [every encoder layer has its own output channels]
    # NOTE: central_block: (C' x H' x W') -> (C' x H' x W') [on the output of the last encoder layer]
    # NOTE: decoder: (C' x H' x W') -> (out_ch x H x W)
    def __init__(
        self, 
        out_ch: int,
        layer_ch: Sequence[int], 
        layer_up: Sequence[int],
        upsampling: Literal["transposed", "bilinear", "nearest", "bicubic"],
        central_block: Callable[..., torch.nn.Module],
        central_block_params: dict[str, Any],
        conv_block: Callable[..., torch.nn.Module],
        conv_block_params: dict[str, Any],
    ):
        super().__init__()
        assert isinstance(layer_ch, Sequence) # (conv_1_ch, conv_2_ch, conv_3_ch, conv_4_ch, conv_5_ch)
        assert isinstance(layer_up, Sequence) # (conv_1_up, conv_2_up, conv_3_up, conv_4_up, conv_5_up)
        assert len(layer_ch) == len(layer_up)
        assert callable(central_block)
        assert isinstance(central_block_params, dict)
        assert callable(conv_block) 
        assert isinstance(conv_block_params, dict)

        # print(f"central_block: in_ch = {layer_ch[-1]}, out_ch = {layer_ch[-1]}")
        self.central_block = central_block(in_ch = layer_ch[-1], out_ch = layer_ch[-1], **central_block_params) 

        # print(f"conv_0: in_ch = {layer_ch[0]}, out_ch = {out_ch}")
        self.conv_0 = conv_block(in_ch = layer_ch[0], out_ch = out_ch, **conv_block_params)

        layer_ch = [layer_ch[0], *layer_ch] # (conv_1_ch, conv_1_ch, conv_2_ch, conv_3_ch, conv_4_ch, conv_5_ch)
        for idx in range(1, len(layer_ch)): # idx = (1, 2, 3, 4, 5)
            # print(f"conv_{idx}: in_ch / skip_ch = {layer_ch[idx]}, out_ch = {layer_ch[idx-1]}, upsampling_factor: {layer_up[idx-1]}")
            setattr(self, f"conv_{idx}", UNetDecoderBlock(
                in_ch=layer_ch[idx], out_ch=layer_ch[idx-1], upsampling=upsampling, upsampling_factor=layer_up[idx-1], 
                conv_block=conv_block, conv_block_params=conv_block_params)
            ) 
            # self.conv_0 = UNetDecoderBlock(in_ch = conv_1_ch, out_ch = out_ch)
            # self.conv_1 = UNetDecoderBlock(in_ch = conv_1_ch, out_ch = conv_1_ch)
            # self.conv_2 = UNetDecoderBlock(in_ch = conv_2_ch, out_ch = conv_1_ch) 
            # self.conv_3 = UNetDecoderBlock(in_ch = conv_3_ch, out_ch = conv_2_ch) 
            # self.conv_4 = UNetDecoderBlock(in_ch = conv_4_ch, out_ch = conv_3_ch) 
            # self.conv_5 = UNetDecoderBlock(in_ch = conv_5_ch, out_ch = conv_4_ch) 
        
    def forward(self, *args):
        # args = x_5, x_4, x_3, x_2, x_1
        x = self.central_block(args[0]) # x = self.central_block(x_5)
        # print("central_block out: ", x.shape)
        for idx in range(len(args), 0, -1): # idx = (5, 4, 3, 2, 1)
            # print(f"conv_{idx} in", "x:", x.shape, f"skip_x ({-idx}):", args[-idx].shape)
            x = getattr(self, f"conv_{idx}")(x, args[-idx]) # x = self.conv_{idx}(x, x_{idx})
            # print(f"conv_{idx} out", x.shape)
        return self.conv_0(x)