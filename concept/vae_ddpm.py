"""A simple VAE + DDPM composition using diffusion in latent space."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .ddpm import DDPM
from .vae import SimpleVAE


class LatentDDPM(nn.Module):
    """Train a DDPM over VAE latent vectors and decode generated latents."""

    def __init__(self, vae: SimpleVAE, diffusion: DDPM) -> None:
        super().__init__()
        self.vae = vae
        self.diffusion = diffusion

    def freeze_vae(self) -> None:
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad = False

    def encode_images(self, images: Tensor, sample: bool = False) -> Tensor:
        mu, logvar = self.vae.encode(images)
        if sample:
            return self.vae.reparameterize(mu, logvar)
        return mu

    def training_loss(self, images: Tensor, detach_vae: bool = True) -> Tensor:
        if detach_vae:
            with torch.no_grad():
                latents = self.encode_images(images)
        else:
            latents = self.encode_images(images)
        return self.diffusion.training_loss(latents)

    @torch.no_grad()
    def sample(self, num_samples: int, latent_dim: int, device: torch.device | str, clamp: bool = True) -> Tensor:
        latents = self.diffusion.sample((num_samples, latent_dim), device=device)
        images = self.vae.decode(latents)
        if clamp:
            images = images.clamp(-1.0, 1.0)
        return images
