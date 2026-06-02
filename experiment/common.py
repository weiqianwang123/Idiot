"""Shared helpers for CIFAR-10 experiments."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "experiment" / "config"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_mode_config(config_path: str | Path, mode: str, path_keys: tuple[str, ...] = ()) -> SimpleNamespace:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if mode not in raw:
        raise KeyError(f"Config {path} does not contain a {mode!r} section.")

    config = {}
    config.update(raw.get("model", {}))
    config.update(raw[mode])
    config["mode"] = mode
    config["config_path"] = path

    for key in path_keys:
        if key in config and config[key] is not None:
            config[key] = resolve_project_path(config[key])

    return SimpleNamespace(**config)


def serializable_args(args: object) -> dict[str, object]:
    clean_args = {}
    for key, value in vars(args).items():
        clean_args[key] = str(value) if isinstance(value, Path) else value
    return clean_args


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto", require_cuda: bool = False) -> torch.device:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but PyTorch cannot use CUDA. "
            "Check the NVIDIA driver and install a CUDA-compatible PyTorch build."
        )
    if require_cuda and device.type != "cuda":
        raise RuntimeError("GPU training requires a CUDA device. Pass --device cuda or fix the CUDA environment.")
    return device


def cifar10_loader(
    data_dir: Path,
    batch_size: int,
    train: bool = True,
    num_workers: int = 2,
    download: bool = True,
) -> DataLoader:
    from torchvision import datasets, transforms

    transform_steps = []
    if train:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=train,
        download=download,
        transform=transforms.Compose(transform_steps),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(images, str(path), nrow=nrow)
