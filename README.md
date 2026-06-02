# Idiot

A personal algorithm playground for manually implementing concepts and running lightweight experiments.

## Structure

- `concept/`: handwritten algorithm and data-structure implementations.
- `experiment/`: config-driven training and evaluation entrypoints for `concept/`.
- `test/`: automatic checks for the implemented concepts.
- `data/`: small sample datasets or input files for experiments.
- `output/`: temporary experiment outputs such as checkpoints and generated images.
- `codex/`: project memory and future Codex-specific workflow notes.

## Environment

The local conda environment for this project is named `idiot`.

```bash
conda activate idiot
pip install -r requirements.txt
```

The environment is pinned to PyTorch CUDA 12.8 wheels:

- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`

This matches the current machine driver reported by `nvidia-smi`: NVIDIA driver `570.172.18`, CUDA `12.8`.

To recreate it from scratch:

```bash
conda env create -f environment.yml
```

## Implemented Algorithms

- `SimpleVAE`: a compact convolutional variational autoencoder for 32x32 RGB images.
- `DDPM`: a minimal denoising diffusion probabilistic model wrapper.
- `DenoiseUNet`: a small image-space noise predictor for CIFAR-10 DDPM experiments.
- `DenoiseDiT`: a compact Diffusion Transformer image-space noise predictor.
- `LatentDenoiser`: an MLP noise predictor for vector latents.
- `LatentDDPM`: a VAE + DDPM combination that diffuses in VAE latent space and decodes samples back to images.

## Experiments

CIFAR-10 is loaded directly through `torchvision.datasets.CIFAR10`. Downloaded data goes under `data/`, and generated samples/checkpoints go under `output/`. Both folders are ignored by git except for their README files.

Automatic GPU checks without downloading data:

```bash
pytest test
```

Train the VAE on GPU:

```bash
python experiment/vae.py train --config experiment/config/vae.json
```

Train the VAE with the KL term disabled:

```bash
python experiment/vae.py train --config experiment/config/vae_no_kl.json
```

Evaluate the trained VAE on the CIFAR-10 test split:

```bash
python experiment/vae.py eval --config experiment/config/vae.json
```

This writes reconstruction pairs and generated samples under `output/vae_eval/`.

Train an image-space DDPM on GPU:

```bash
python experiment/ddpm.py train --config experiment/config/ddpm_unet.json
```

Train an image-space DDPM with DiT denoising:

```bash
python experiment/ddpm.py train --config experiment/config/ddpm_dit.json
```

Evaluate the trained image-space DDPM:

```bash
python experiment/ddpm.py eval --config experiment/config/ddpm_unet.json
```

Evaluate the trained DiT image-space DDPM:

```bash
python experiment/ddpm.py eval --config experiment/config/ddpm_dit.json
```

This reports test-set denoising MSE and writes generated samples under the eval `output_dir` set by the config.

Train the VAE + latent DDPM combination on GPU after a VAE checkpoint exists:

```bash
python experiment/latent_ddpm.py train --config experiment/config/latent_ddpm.json
```

Evaluate the trained latent DDPM:

```bash
python experiment/latent_ddpm.py eval --config experiment/config/latent_ddpm.json
```

This reports latent denoising MSE and writes decoded generated samples under `output/latent_ddpm_eval/`.

## Guiding Principles

- Keep implementations readable and explicit.
- Prefer small focused files over broad frameworks.
- Use `experiment/` to validate behavior with simple, repeatable checks.
- Keep sample data tiny unless a larger file is specifically needed.
