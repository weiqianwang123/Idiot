"""Config-driven pixel-space DDPM train/eval entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from concept import DDPM, DenoiseDiT, DenoiseUNet
from experiment.common import CONFIG_DIR, cifar10_loader, get_device, load_mode_config, save_image_grid, seed_all, serializable_args


PATH_KEYS = ("data_dir", "output_dir", "checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "ddpm_unet.json")
    return parser.parse_args()


def checkpoint_value(checkpoint: dict[str, object], key: str, fallback: object) -> object:
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, dict):
        return saved_args.get(key, fallback)
    return fallback


def build_denoiser(args: argparse.Namespace, checkpoint: dict[str, object] | None = None) -> tuple[torch.nn.Module, str]:
    checkpoint = checkpoint or {}
    denoiser_name = str(getattr(args, "denoiser", checkpoint_value(checkpoint, "denoiser", "unet")))
    if denoiser_name == "unet":
        base_channels = int(getattr(args, "base_channels", checkpoint_value(checkpoint, "base_channels", 64)))
        return DenoiseUNet(base_channels=base_channels), denoiser_name

    patch_size = int(getattr(args, "dit_patch_size", checkpoint_value(checkpoint, "dit_patch_size", 4)))
    hidden_dim = int(getattr(args, "dit_hidden_dim", checkpoint_value(checkpoint, "dit_hidden_dim", 256)))
    depth = int(getattr(args, "dit_depth", checkpoint_value(checkpoint, "dit_depth", 6)))
    heads = int(getattr(args, "dit_heads", checkpoint_value(checkpoint, "dit_heads", 8)))
    mlp_ratio = float(getattr(args, "dit_mlp_ratio", checkpoint_value(checkpoint, "dit_mlp_ratio", 4.0)))
    return (
        DenoiseDiT(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
        ),
        denoiser_name,
    )


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = get_device(args.device, require_cuda=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = cifar10_loader(args.data_dir, args.batch_size, train=True, num_workers=args.num_workers)
    denoiser, _ = build_denoiser(args)
    diffusion = DDPM(denoiser, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"{args.denoiser} ddpm epoch {epoch}/{args.epochs}")
        for images, _ in progress:
            images = images.to(device)
            loss = diffusion.training_loss(images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(len(loader), 1)
        checkpoint = {
            "model": diffusion.state_dict(),
            "args": serializable_args(args),
            "epoch": epoch,
            "loss": mean_loss,
        }
        torch.save(checkpoint, args.output_dir / "ddpm.pt")

        diffusion.eval()
        with torch.no_grad():
            samples = diffusion.sample((args.sample_count, 3, 32, 32), device=device)
            save_image_grid(samples, args.output_dir / f"samples_epoch_{epoch}.png")

        print(f"epoch={epoch} loss={mean_loss:.6f} checkpoint={args.output_dir / 'ddpm.pt'}")


def evaluate(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    timesteps = int(getattr(args, "timesteps", checkpoint_value(checkpoint, "timesteps", 1000)))

    denoiser, denoiser_name = build_denoiser(args, checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diffusion = DDPM(denoiser, timesteps=timesteps).to(device)
    diffusion.load_state_dict(checkpoint["model"])
    diffusion.eval()

    loader = cifar10_loader(args.data_dir, args.batch_size, train=False, num_workers=args.num_workers)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="eval pixel ddpm")
        for batch_index, (images, _) in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            images = images.to(device)
            batch_loss = 0.0
            for _ in range(args.loss_repeats):
                batch_loss += diffusion.training_loss(images).item()
            batch_loss /= args.loss_repeats

            total_loss += batch_loss
            total_batches += 1
            progress.set_postfix(denoise_mse=f"{batch_loss:.4f}")

        if total_batches == 0:
            raise RuntimeError("No batches were evaluated. Check max_batches and the CIFAR-10 loader.")

        sample_path = None
        if not args.skip_sampling:
            samples = diffusion.sample((args.sample_count, 3, 32, 32), device=device).detach().cpu()
            sample_path = args.output_dir / "samples.png"
            save_image_grid(samples, sample_path)

    print(f"checkpoint={args.checkpoint}")
    print(f"batches={total_batches}")
    print(f"denoising_mse={total_loss / total_batches:.6f}")
    print(f"denoiser={denoiser_name}")
    print(f"timesteps={timesteps}")
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
