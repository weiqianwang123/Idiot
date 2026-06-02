"""A compact Diffusion Transformer denoiser for 32x32 images."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .ddpm import sinusoidal_time_embedding


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with timestep-conditioned adaptive layer norm."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.ada_norm = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_norm(condition).chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_output

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class DenoiseDiT(nn.Module):
    """Small DiT noise predictor for CIFAR-10 sized DDPM experiments."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        image_channels: int = 3,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.image_size = image_size
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side**2

        self.patch_embed = nn.Conv2d(image_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_ada_norm = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.final_proj = nn.Linear(hidden_dim, patch_size * patch_size * image_channels)

        nn.init.normal_(self.pos_embed, std=0.02)

    def patchify(self, images: Tensor) -> Tensor:
        tokens = self.patch_embed(images)
        return tokens.flatten(2).transpose(1, 2)

    def unpatchify(self, patches: Tensor) -> Tensor:
        batch_size = patches.size(0)
        patch = self.patch_size
        channels = self.image_channels
        side = self.num_patches_per_side
        patches = patches.view(batch_size, side, side, patch, patch, channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, channels, self.image_size, self.image_size)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        tokens = self.patchify(x) + self.pos_embed
        condition = sinusoidal_time_embedding(timesteps, self.hidden_dim)
        condition = self.time_mlp(condition)

        for block in self.blocks:
            tokens = block(tokens, condition)

        shift, scale = self.final_ada_norm(condition).chunk(2, dim=1)
        tokens = modulate(self.final_norm(tokens), shift, scale)
        patches = self.final_proj(tokens)
        return self.unpatchify(patches)
