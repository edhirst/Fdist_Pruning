"""
SimpleViT: a compact vision transformer for small images, plus its parts.

Written out with plain nn.Linear layers and an explicit softmax attention rather
than nn.MultiheadAttention, so that torch.func per-sample gradients work on it
and every learnable tensor is one the pruners can see. See the class docstrings
below for why each choice was made.
"""
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention implemented with plain nn.Linear layers
    and an explicit softmax(QK^T/sqrt(d))V.

    Deliberately avoids nn.MultiheadAttention / F.scaled_dot_product_attention so
    that torch.func (vmap/grad/functional_call) per-sample gradients used by the
    backprop Fisher calculator work reliably on CPU/MPS, and so that all learnable
    tensors are ordinary Linear weights/biases as seen by the pruners.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return self.proj_drop(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleViT(nn.Module):
    """
    Compact Vision Transformer for small images (CIFAR-10 / MNIST scale),
    in the ViT-Lite style (Hassani et al., arXiv:2104.05704):

        Conv2d patchify -> learned positional embedding -> depth x pre-LN blocks
        -> final LayerNorm -> mean-pool over tokens -> Linear head

    Mean pooling (no CLS token) follows Beyer et al.'s simple-ViT baseline.
    Defaults (patch 4, dim 128, depth 4, 4 heads, mlp_ratio 2) give ~546k
    parameters on CIFAR-10 (64 tokens); on 28x28 inputs there are 49 tokens.
    """

    def __init__(self, img_size=32, in_channels=3, patch_size=4, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=2.0, num_classes=10, dropout=0.0):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        num_patches = (img_size // patch_size) ** 2

        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)            # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)                  # mean-pool over tokens
        return self.head(x)
