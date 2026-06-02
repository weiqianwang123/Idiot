# Experiment

Config-driven training and evaluation entrypoints for implementations in `concept/` live here.

## Structure

- `config/`: model configs with `model`, `train`, and `eval` sections.
- `vae.py`: VAE train/eval entrypoint.
- `ddpm.py`: pixel-space DDPM train/eval entrypoint for U-Net or DiT denoisers.
- `latent_ddpm.py`: VAE latent DDPM train/eval entrypoint.
- `common.py`: shared CIFAR-10 loading, device, output, and serialization helpers.

## Current Configs

- `config/vae.json`: regular VAE.
- `config/vae_no_kl.json`: VAE with KL disabled.
- `config/ddpm_unet.json`: pixel-space DDPM with U-Net denoiser.
- `config/ddpm_dit.json`: pixel-space DDPM with DiT denoiser.
- `config/latent_ddpm.json`: DDPM over VAE latents.

Run scripts from the repository root after activating the conda environment:

```bash
conda activate idiot
python experiment/vae.py train --config experiment/config/vae.json
python experiment/vae.py train --config experiment/config/vae_no_kl.json
python experiment/vae.py eval --config experiment/config/vae.json
python experiment/ddpm.py train --config experiment/config/ddpm_dit.json
python experiment/ddpm.py eval --config experiment/config/ddpm_dit.json
python experiment/latent_ddpm.py eval --config experiment/config/latent_ddpm.json
```

Training configs default to `device: "cuda"` and fail early if CUDA is unavailable.
