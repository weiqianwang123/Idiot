"""Automatic checks for the generative model concepts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concept import DDPM, DenoiseDiT, DenoiseUNet, LatentDDPM, LatentDenoiser, SimpleVAE, vae_loss


@pytest.fixture()
def device() -> torch.device:
    assert torch.cuda.is_available(), "CUDA is required for GPU training checks."
    return torch.device("cuda")


def test_cuda_device_is_available(device: torch.device) -> None:
    assert torch.cuda.get_device_name(device)


def test_simple_vae_forward_and_loss(device: torch.device) -> None:
    images = torch.randn(2, 3, 32, 32, device=device).clamp(-1.0, 1.0)
    vae = SimpleVAE(latent_dim=16, hidden_channels=16).to(device)

    recon, mu, logvar = vae(images)
    loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=1e-4)

    assert recon.shape == images.shape
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
    assert loss.ndim == 0
    assert recon_loss.ndim == 0
    assert kl_loss.ndim == 0


def test_image_ddpm_training_loss(device: torch.device) -> None:
    images = torch.randn(2, 3, 32, 32, device=device).clamp(-1.0, 1.0)
    diffusion = DDPM(DenoiseUNet(base_channels=16), timesteps=4).to(device)

    loss = diffusion.training_loss(images)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dit_ddpm_training_loss(device: torch.device) -> None:
    images = torch.randn(2, 3, 32, 32, device=device).clamp(-1.0, 1.0)
    denoiser = DenoiseDiT(patch_size=4, hidden_dim=32, depth=2, num_heads=4)
    diffusion = DDPM(denoiser, timesteps=4).to(device)

    loss = diffusion.training_loss(images)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_latent_ddpm_training_and_sample(device: torch.device) -> None:
    images = torch.randn(2, 3, 32, 32, device=device).clamp(-1.0, 1.0)
    vae = SimpleVAE(latent_dim=16, hidden_channels=16).to(device)
    latent_diffusion = DDPM(LatentDenoiser(latent_dim=16, hidden_dim=32), timesteps=4).to(device)
    latent_model = LatentDDPM(vae, latent_diffusion).to(device)

    loss = latent_model.training_loss(images)
    with torch.no_grad():
        samples = latent_model.sample(num_samples=2, latent_dim=16, device=device)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert samples.shape == images.shape
