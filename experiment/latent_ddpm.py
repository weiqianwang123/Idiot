"""Config-driven latent DDPM train/eval entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from concept import DDPM, LatentDDPM, LatentDenoiser, SimpleVAE
from experiment.common import CONFIG_DIR, cifar10_loader, get_device, load_mode_config, save_image_grid, seed_all, serializable_args


PATH_KEYS = ("data_dir", "output_dir", "checkpoint", "vae_checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "latent_ddpm.json")
    return parser.parse_args()


def checkpoint_value(checkpoint: dict[str, object], key: str, fallback: object) -> object:
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, dict):
        return saved_args.get(key, fallback)
    return fallback


def load_vae(checkpoint_path: Path, latent_dim: int, hidden_channels: int, device: torch.device) -> SimpleVAE:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {checkpoint_path}. "
            "Run python experiment/vae.py train --config experiment/config/vae.json first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    vae = SimpleVAE(latent_dim=latent_dim, hidden_channels=hidden_channels).to(device)
    vae.load_state_dict(checkpoint["model"])
    vae.eval()
    return vae


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = get_device(args.device, require_cuda=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = cifar10_loader(args.data_dir, args.batch_size, train=True, num_workers=args.num_workers)
    vae = load_vae(args.vae_checkpoint, args.latent_dim, args.hidden_channels, device)
    denoiser = LatentDenoiser(latent_dim=args.latent_dim, hidden_dim=args.latent_hidden_dim)
    diffusion = DDPM(denoiser, timesteps=args.timesteps).to(device)
    model = LatentDDPM(vae, diffusion).to(device)
    model.freeze_vae()

    optimizer = torch.optim.AdamW(model.diffusion.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.diffusion.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"latent ddpm epoch {epoch}/{args.epochs}")
        for images, _ in progress:
            images = images.to(device)
            loss = model.training_loss(images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(len(loader), 1)
        checkpoint = {
            "model": model.diffusion.state_dict(),
            "args": serializable_args(args),
            "epoch": epoch,
            "loss": mean_loss,
        }
        torch.save(checkpoint, args.output_dir / "latent_ddpm.pt")

        model.eval()
        with torch.no_grad():
            samples = model.sample(args.sample_count, latent_dim=args.latent_dim, device=device)
            save_image_grid(samples, args.output_dir / f"samples_epoch_{epoch}.png")

        print(f"epoch={epoch} loss={mean_loss:.6f} checkpoint={args.output_dir / 'latent_ddpm.pt'}")


def evaluate(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    latent_dim = int(getattr(args, "latent_dim", checkpoint_value(checkpoint, "latent_dim", 128)))
    hidden_channels = int(getattr(args, "hidden_channels", checkpoint_value(checkpoint, "hidden_channels", 64)))
    latent_hidden_dim = int(getattr(args, "latent_hidden_dim", checkpoint_value(checkpoint, "latent_hidden_dim", 512)))
    timesteps = int(getattr(args, "timesteps", checkpoint_value(checkpoint, "timesteps", 1000)))

    vae_checkpoint = getattr(args, "vae_checkpoint", None)
    if vae_checkpoint is None:
        vae_checkpoint = Path(str(checkpoint_value(checkpoint, "vae_checkpoint", PROJECT_ROOT / "output" / "vae" / "vae.pt")))

    vae = load_vae(vae_checkpoint, latent_dim=latent_dim, hidden_channels=hidden_channels, device=device)
    denoiser = LatentDenoiser(latent_dim=latent_dim, hidden_dim=latent_hidden_dim)
    diffusion = DDPM(denoiser, timesteps=timesteps).to(device)
    diffusion.load_state_dict(checkpoint["model"])
    model = LatentDDPM(vae, diffusion).to(device)
    model.freeze_vae()
    model.eval()

    loader = cifar10_loader(args.data_dir, args.batch_size, train=False, num_workers=args.num_workers)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="eval latent ddpm")
        for batch_index, (images, _) in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            images = images.to(device)
            latents = model.encode_images(images)

            batch_loss = 0.0
            for _ in range(args.loss_repeats):
                batch_loss += model.diffusion.training_loss(latents).item()
            batch_loss /= args.loss_repeats

            total_loss += batch_loss
            total_batches += 1
            progress.set_postfix(latent_denoise_mse=f"{batch_loss:.4f}")

        if total_batches == 0:
            raise RuntimeError("No batches were evaluated. Check max_batches and the CIFAR-10 loader.")

        sample_path = None
        if not args.skip_sampling:
            samples = model.sample(args.sample_count, latent_dim=latent_dim, device=device).detach().cpu()
            sample_path = args.output_dir / "samples.png"
            save_image_grid(samples, sample_path)

    print(f"checkpoint={args.checkpoint}")
    print(f"vae_checkpoint={vae_checkpoint}")
    print(f"batches={total_batches}")
    print(f"latent_denoising_mse={total_loss / total_batches:.6f}")
    print(f"timesteps={timesteps}")
    print(f"latent_dim={latent_dim}")
    print(f"latent_hidden_dim={latent_hidden_dim}")
    if sample_path is not None:
        print(f"samples={sample_path}")


def main() -> None:
    cli_args = parse_args()
    config = load_mode_config(cli_args.config, cli_args.mode, path_keys=PATH_KEYS)
    if cli_args.mode == "train":
        train(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
