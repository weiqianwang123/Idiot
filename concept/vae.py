"""A compact convolutional VAE for 32x32 RGB images."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SimpleVAE(nn.Module):
    """Small VAE aimed at CIFAR-10 scale images.

    Inputs and outputs are expected to be in the `[-1, 1]` range.
    """

    def __init__(self, image_channels: int = 3, latent_dim: int = 128, hidden_channels: int = 64) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.SiLU(),
        )

        encoded_dim = hidden_channels * 4 * 4 * 4
        self.to_mu = nn.Linear(encoded_dim, latent_dim)
        self.to_logvar = nn.Linear(encoded_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, encoded_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.from_latent(z)
        h = h.view(z.size(0), -1, 4, 4)
        return self.decoder(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    """Return total VAE loss plus reconstruction and KL terms."""

    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
