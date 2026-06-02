"""Minimal DDPM components for images and latent vectors."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Create transformer-style sinusoidal embeddings for integer timesteps."""

    half_dim = dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _extract(values: Tensor, timesteps: Tensor, target_shape: torch.Size) -> Tensor:
    """Gather one scalar per batch item and reshape for broadcasting."""

    batch_size = timesteps.shape[0]
    out = values.gather(0, timesteps)
    return out.reshape(batch_size, *((1,) * (len(target_shape) - 1)))


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Small residual block conditioned on a time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DenoiseUNet(nn.Module):
    """A deliberately small U-Net noise predictor for 32x32 images."""

    def __init__(self, image_channels: int = 3, base_channels: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_block = ResidualBlock(image_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down1_block = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.down2_block = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.mid_block = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1_block = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2_block = ResidualBlock(base_channels * 2, base_channels, time_dim)

        self.out = nn.Sequential(
            _group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        time_emb = sinusoidal_time_embedding(timesteps, self.time_dim)
        time_emb = self.time_mlp(time_emb)

        h0 = self.in_block(x, time_emb)
        h1 = self.down1_block(self.down1(h0), time_emb)
        h2 = self.down2_block(self.down2(h1), time_emb)

        h = self.mid_block(h2, time_emb)
        h = self.up1(h)
        h = self.up1_block(torch.cat([h, h1], dim=1), time_emb)
        h = self.up2(h)
        h = self.up2_block(torch.cat([h, h0], dim=1), time_emb)
        return self.out(h)


class LatentDenoiser(nn.Module):
    """MLP noise predictor for VAE latent vectors."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: Tensor, timesteps: Tensor) -> Tensor:
        time_emb = sinusoidal_time_embedding(timesteps, self.time_dim)
        return self.net(torch.cat([z, time_emb], dim=1))


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapper.

    The denoise model must accept `(x_t, timesteps)` and predict the noise
    that was added to the clean sample.
    """

    def __init__(
        self,
        denoise_model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars))

    def q_sample(self, x_start: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape) * noise
        )

    def predict_noise(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        return self.denoise_model(x_t, timesteps)

    def training_loss(self, x_start: Tensor) -> Tensor:
        batch_size = x_start.size(0)
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        predicted_noise = self.predict_noise(x_t, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def forward(self, x_start: Tensor) -> Tensor:
        return self.training_loss(x_start)

    @torch.no_grad()
    def p_sample(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        betas_t = _extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus_alpha_bars_t = _extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, timesteps, x_t.shape)

        predicted_noise = self.predict_noise(x_t, timesteps)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_bars_t)

        noise = torch.randn_like(x_t)
        posterior_variance_t = _extract(self.posterior_variance, timesteps, x_t.shape)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.size(0), *((1,) * (x_t.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps)
        return x
