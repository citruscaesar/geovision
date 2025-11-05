from typing import Optional, Literal, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from geovision.models import get_state_dict


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition input into non-overlapping windows
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Merge windows back to feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    """Multi-Layer Perceptron with GELU activation"""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention (W-MSA) module with relative position bias"""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionV2(nn.Module):
    """Window-based Multi-head Self Attention with cosine attention (Swin V2)"""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 pretrained_window_size: tuple[int, int] = (0, 0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Relative position index
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        # Continuous relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(self, dim: int, input_resolution: tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, use_v2: bool = False,
                 pretrained_window_size: int = 0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)

        if use_v2:
            self.attn = WindowAttentionV2(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                pretrained_window_size=(pretrained_window_size, pretrained_window_size))
        else:
            self.attn = WindowAttention(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer - downsamples by 2x"""

    def __init__(self, input_resolution: tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""

    def __init__(self, dim: int, input_resolution: tuple[int, int], depth: int,
                 num_heads: int, window_size: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float | list[float] = 0.0, downsample: Optional[nn.Module] = None,
                 use_v2: bool = False, pretrained_window_size: int = 0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_v2=use_v2, pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class SwinTransformer_Encoder(nn.Module):
    """Swin Transformer Encoder

    Supports both Swin Transformer V1 and V2 architectures.
    V2 uses scaled cosine attention and continuous relative position bias.
    """

    swin_configs = {
        "tiny": {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
        "small": {"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]},
        "base": {"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]},
        "large": {"embed_dim": 192, "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48]},
    }

    def __init__(
        self,
        model_size: Literal["tiny", "small", "base", "large"] = "tiny",
        img_size: int = 224,
        patch_size: int = 4,
        input_channels: int = 3,
        embed_dim: Optional[int] = None,
        depths: Optional[Sequence[int]] = None,
        num_heads: Optional[Sequence[int]] = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_v2: bool = False,  # Set to True for Swin V2
        pretrained_window_size: int = 0,  # For Swin V2 pretrained models
        weights_init: Literal["random", "torchvision", "torchgeo", "url", "path"] = "random",
        weights_param: Optional[str] = None,
    ):
        """
        Initialize Swin Transformer Encoder

        Parameters:
        -----------
        model_size : str
            One of ["tiny", "small", "base", "large"]
        img_size : int
            Input image size (default: 224)
        patch_size : int
            Patch size (default: 4)
        input_channels : int
            Number of input channels (default: 3)
        embed_dim : int, optional
            Patch embedding dimension. Overrides model_size if provided.
        depths : Sequence[int], optional
            Depth of each Swin Transformer stage. Overrides model_size if provided.
        num_heads : Sequence[int], optional
            Number of attention heads in each stage. Overrides model_size if provided.
        window_size : int
            Window size (default: 7)
        mlp_ratio : float
            Ratio of mlp hidden dim to embedding dim (default: 4.0)
        qkv_bias : bool
            If True, add a learnable bias to query, key, value (default: True)
        drop_rate : float
            Dropout rate (default: 0.0)
        attn_drop_rate : float
            Attention dropout rate (default: 0.0)
        drop_path_rate : float
            Stochastic depth rate (default: 0.1)
        use_v2 : bool
            Whether to use Swin Transformer V2 (default: False)
        pretrained_window_size : int
            Window size used in pretraining (for Swin V2) (default: 0)
        weights_init : str
            Weight initialization method
        weights_param : str, optional
            Parameter for weight initialization
        """
        super().__init__()

        # Use predefined config or custom parameters
        if embed_dim is None or depths is None or num_heads is None:
            assert model_size in self.swin_configs, \
                f"model_size must be one of {list(self.swin_configs.keys())}, got {model_size}"
            config = self.swin_configs[model_size]
            embed_dim = embed_dim or config["embed_dim"]
            depths = depths or config["depths"]
            num_heads = num_heads or config["num_heads"]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self._num_input_channels = input_channels
        self._out_ch_per_layer = []
        self._downsampling_per_layer = []

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=input_channels, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Track for interface compatibility
        self._out_ch_per_layer.append(embed_dim)
        self._downsampling_per_layer.append(patch_size)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_v2=use_v2,
                pretrained_window_size=pretrained_window_size)
            self.layers.append(layer)

            # Track output channels and downsampling
            self._out_ch_per_layer.append(int(embed_dim * 2 ** i_layer))
            if i_layer < self.num_layers - 1:
                self._downsampling_per_layer.append(2)
            else:
                self._downsampling_per_layer.append(1)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        # Initialize weights
        self._init_weights()

        # Load pretrained weights if specified
        if weights_init != "random":
            state_dict = get_state_dict(weights_init, weights_param)
            self.load_state_dict(state_dict, strict=False)

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning final stage features"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning features from all stages (for dense prediction)"""
        features = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features


# Convenience constructors for Swin Transformer V1
def swin_tiny(img_size: int = 224, input_channels: int = 3, **kwargs):
    """Swin-T (Tiny) - 28M params"""
    return SwinTransformer_Encoder(
        model_size="tiny", img_size=img_size, input_channels=input_channels, use_v2=False, **kwargs)


def swin_small(img_size: int = 224, input_channels: int = 3, **kwargs):
    """Swin-S (Small) - 50M params"""
    return SwinTransformer_Encoder(
        model_size="small", img_size=img_size, input_channels=input_channels, use_v2=False, **kwargs)


def swin_base(img_size: int = 224, input_channels: int = 3, **kwargs):
    """Swin-B (Base) - 88M params"""
    return SwinTransformer_Encoder(
        model_size="base", img_size=img_size, input_channels=input_channels, use_v2=False, **kwargs)


def swin_large(img_size: int = 224, input_channels: int = 3, **kwargs):
    """Swin-L (Large) - 197M params"""
    return SwinTransformer_Encoder(
        model_size="large", img_size=img_size, input_channels=input_channels, use_v2=False, **kwargs)


# Convenience constructors for Swin Transformer V2
def swinv2_tiny(img_size: int = 256, input_channels: int = 3, window_size: int = 16, **kwargs):
    """Swin V2-T (Tiny) with scaled cosine attention"""
    return SwinTransformer_Encoder(
        model_size="tiny", img_size=img_size, input_channels=input_channels,
        window_size=window_size, use_v2=True, **kwargs)


def swinv2_small(img_size: int = 256, input_channels: int = 3, window_size: int = 16, **kwargs):
    """Swin V2-S (Small) with scaled cosine attention"""
    return SwinTransformer_Encoder(
        model_size="small", img_size=img_size, input_channels=input_channels,
        window_size=window_size, use_v2=True, **kwargs)


def swinv2_base(img_size: int = 256, input_channels: int = 3, window_size: int = 16, **kwargs):
    """Swin V2-B (Base) with scaled cosine attention"""
    return SwinTransformer_Encoder(
        model_size="base", img_size=img_size, input_channels=input_channels,
        window_size=window_size, use_v2=True, **kwargs)


def swinv2_large(img_size: int = 256, input_channels: int = 3, window_size: int = 16, **kwargs):
    """Swin V2-L (Large) with scaled cosine attention"""
    return SwinTransformer_Encoder(
        model_size="large", img_size=img_size, input_channels=input_channels,
        window_size=window_size, use_v2=True, **kwargs)
