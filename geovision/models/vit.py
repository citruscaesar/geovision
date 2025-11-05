from typing import Optional, Literal
import math

import torch
import torch.nn as nn
from collections import OrderedDict

from geovision.models import get_state_dict


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        weights_init: Literal["random", "torchvision", "torchgeo", "url", "path"] = "random",
        weights_param: Optional[str] = None
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Load pretrained weights if specified
        if weights_init != "random":
            self.load_state_dict(get_state_dict(weights_init, weights_param), strict=False)
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        x = self.blocks(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Return class token representation
        return x[:, 0]  # (B, embed_dim)


# Predefined ViT configurations
def vit_tiny_patch16_224(**kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs
    )


def vit_small_patch16_224(**kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs
    )


def vit_base_patch16_224(**kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )


def vit_large_patch16_224(**kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )


def vit_huge_patch14_224(**kwargs):
    return VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs
    )


class ViTEncoder(nn.Module):
    """ViT encoder that follows the project's encoder interface"""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        weights_init: Literal["random", "torchvision", "torchgeo", "url", "path"] = "random",
        weights_param: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # Model configurations
        model_configs = {
            "vit_tiny_patch16_224": vit_tiny_patch16_224,
            "vit_small_patch16_224": vit_small_patch16_224,
            "vit_base_patch16_224": vit_base_patch16_224,
            "vit_large_patch16_224": vit_large_patch16_224,
            "vit_huge_patch14_224": vit_huge_patch14_224,
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_configs.keys())}")
        
        # Create model
        self.model = model_configs[model_name](
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            weights_init=weights_init,
            weights_param=weights_param,
            **kwargs
        )
        
        # Set attributes for compatibility with existing interface
        self._out_ch_per_layer = [self.model.embed_dim]  # Only final layer output
        self._downsampling_per_layer = [img_size // patch_size]  # Patch-based downsampling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)