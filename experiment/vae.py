"""Config-driven VAE train/eval entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from concept import SimpleVAE, vae_loss
from experiment.common import CONFIG_DIR, cifar10_loader, get_device, load_mode_config, save_image_grid, seed_all, serializable_args


PATH_KEYS = ("data_dir", "output_dir", "checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "vae.json")
    return parser.parse_args()


def checkpoint_value(checkpoint: dict[str, object], key: str, fallback: object) -> object:
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, dict):
        return saved_args.get(key, fallback)
    return fallback


def train(args: argparse.Namespace) -> None:
    if getattr(args, "disable_kl", False):
        args.beta = 0.0

    seed_all(args.seed)
    device = get_device(args.device, require_cuda=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = cifar10_loader(args.data_dir, args.batch_size, train=True, num_workers=args.num_workers)
    model = SimpleVAE(latent_dim=args.latent_dim, hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"vae epoch {epoch}/{args.epochs}")
        for images, _ in progress:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=args.beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                kl=f"{kl_loss.item():.4f}",
                beta=f"{args.beta:g}",
            )

        mean_loss = running_loss / max(len(loader), 1)
        checkpoint = {
            "model": model.state_dict(),
            "args": serializable_args(args),
            "epoch": epoch,
            "loss": mean_loss,
        }
        torch.save(checkpoint, args.output_dir / "vae.pt")

        model.eval()
        with torch.no_grad():
            samples = model.sample(args.sample_count, device=device)
            save_image_grid(samples, args.output_dir / f"samples_epoch_{epoch}.png")
            recon_images, _, _ = model(images[: args.sample_count])
            save_image_grid(recon_images, args.output_dir / f"recon_epoch_{epoch}.png")

        print(f"epoch={epoch} loss={mean_loss:.6f} checkpoint={args.output_dir / 'vae.pt'}")


def evaluate(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    latent_dim = int(getattr(args, "latent_dim", checkpoint_value(checkpoint, "latent_dim", 128)))
    hidden_channels = int(getattr(args, "hidden_channels", checkpoint_value(checkpoint, "hidden_channels", 64)))
    beta = float(getattr(args, "beta", checkpoint_value(checkpoint, "beta", 1e-4)))

    model = SimpleVAE(latent_dim=latent_dim, hidden_channels=hidden_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = cifar10_loader(args.data_dir, args.batch_size, train=False, num_workers=args.num_workers)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_images = 0
    comparison_images = None
    comparison_recons = None

    with torch.no_grad():
        progress = tqdm(loader, desc="eval vae")
        for batch_index, (images, _) in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=beta)

            batch_size = images.size(0)
            total_images += batch_size
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size

            if comparison_images is None:
                count = min(args.comparison_count, batch_size)
                comparison_images = images[:count].detach().cpu()
                comparison_recons = recon[:count].detach().cpu()

            progress.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", kl=f"{kl_loss.item():.4f}")

        if total_images == 0:
            raise RuntimeError("No images were evaluated. Check max_batches and the CIFAR-10 loader.")

        samples = model.sample(args.sample_count, device=device).detach().cpu()

    paired = torch.stack([comparison_images, comparison_recons], dim=1).flatten(0, 1)
    comparison_path = args.output_dir / "reconstruction_pairs.png"
    samples_path = args.output_dir / "samples.png"
    save_image_grid(paired, comparison_path, nrow=2)
    save_image_grid(samples, samples_path)

    print(f"checkpoint={args.checkpoint}")
    print(f"images={total_images}")
    print(f"loss={total_loss / total_images:.6f}")
    print(f"reconstruction_loss={total_recon / total_images:.6f}")
    print(f"kl_loss={total_kl / total_images:.6f}")
    print(f"reconstruction_pairs={comparison_path}")
    print(f"samples={samples_path}")


def main() -> None:
    cli_args = parse_args()
    config = load_mode_config(cli_args.config, cli_args.mode, path_keys=PATH_KEYS)
    if cli_args.mode == "train":
        train(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
