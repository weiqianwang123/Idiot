# Idiot Project Memory

This repo is a personal algorithm playground.

## Purpose

- Manually implement algorithms and data structures.
- Keep implementations small, readable, and easy to reason about.
- Add lightweight experiments or tests that validate behavior without heavy setup.

## Structure

- `concept/`: handwritten algorithm and data-structure implementations.
- `experiment/`: config-driven training and evaluation entrypoints for `concept/`.
- `test/`: automatic checks for concept implementations.
- `data/`: small sample datasets and input files for experiments.
- `output/`: temporary generated results and checkpoints.
- `codex/skills/`: future local workflow notes or Codex skills for this project.

## Working Style

- Prefer simple, explicit code over clever abstractions.
- Keep each algorithm implementation focused on one idea.
- Put quick validation code in `experiment/`, not inside the core implementation unless it is a tiny self-check.
- When adding tests, include ordinary cases, edge cases, and at least one failure-prone case.

## Current Status

- Conda environment `idiot` created with Python 3.11.
- Environment pinned to PyTorch CUDA 12.8 wheels for GPU training on the current NVIDIA driver.
- VAE, DDPM, DiT denoiser, and VAE + latent DDPM implementations added.
- Experiment configs live in `experiment/config/`.
- VAE train/eval lives in `experiment/vae.py`.
- Pixel-space DDPM train/eval lives in `experiment/ddpm.py`.
- Latent DDPM train/eval lives in `experiment/latent_ddpm.py`.
- VAE no-KL training is controlled by `experiment/config/vae_no_kl.json`.
- Automatic GPU checks live in `test/` and run with `pytest test`.
- `data/` and `output/` contents are ignored by git except README files.
