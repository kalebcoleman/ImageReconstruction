#!/usr/bin/env python3
"""Generate a clearer VAE interpolation panel from visually different endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MNIST_PAIRS = ((1, 8), (3, 9), (0, 6), (2, 7))
FASHION_PAIRS = ((7, 0), (5, 4), (8, 1))
FASHION_LABELS = {
    0: "shirt",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "boot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a better VAE latent interpolation image.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a trained VAE checkpoint.")
    parser.add_argument("--dataset", choices=("mnist", "fashion_mnist", "fashion"), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--scale", type=int, default=4, help="Nearest-neighbor display scale for each decoded tile.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/combined/vae_interpolation_better.png"),
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_placeholder(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 240), "#fffdfa")
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="#b13f32", width=4)
    draw.text((32, 32), "Better VAE interpolation not generated", font=font, fill="#17130f")
    draw.multiline_text((32, 88), message, font=load_font(18), fill="#5f554d", spacing=8)
    image.save(path)


def checkpoint_candidates() -> list[Path]:
    roots = (Path("outputs"), Path("output"), Path("results"), Path("runs"), Path("experiments"), Path("logs"))
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            for suffix in ("*.pt", "*.pth", "*.ckpt"):
                candidates.extend(root.rglob(suffix))
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def checkpoint_payload(path: Path) -> dict[str, Any] | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Cannot inspect checkpoints because PyTorch is unavailable: {exc}")
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def looks_like_vae_checkpoint(payload: dict[str, Any]) -> bool:
    config = payload.get("config")
    if isinstance(config, dict) and config.get("model") == "vae":
        return True
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        return False
    keys = set(state)
    return {"mu.weight", "logvar.weight"}.issubset(keys) and any(key.startswith("decoder.") for key in keys)


def find_checkpoint(explicit: Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    if explicit is not None:
        payload = checkpoint_payload(explicit)
        if payload is not None and looks_like_vae_checkpoint(payload):
            return explicit, payload
        return None, None
    for path in checkpoint_candidates():
        payload = checkpoint_payload(path)
        if payload is not None and looks_like_vae_checkpoint(payload):
            return path, payload
    return None, None


def latent_dim_from_payload(payload: dict[str, Any], fallback: int) -> int:
    config = payload.get("config")
    if isinstance(config, dict):
        value = config.get("latent_dim")
        if isinstance(value, int) and value > 0:
            return value
    state = payload.get("model_state_dict", payload)
    if isinstance(state, dict):
        weight = state.get("mu.weight")
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[0])
    return fallback


def load_dataset(name: str, data_dir: Path):
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    if name in ("fashion", "fashion_mnist"):
        dataset = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform)
        return dataset, FASHION_PAIRS, FASHION_LABELS
    dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    return dataset, MNIST_PAIRS, {index: str(index) for index in range(10)}


def choose_endpoints(dataset: Any, pairs: tuple[tuple[int, int], ...]) -> tuple[Any, int, Any, int]:
    by_label: dict[int, Any] = {}
    for image, label in dataset:
        label = int(label)
        by_label.setdefault(label, image)
        for left, right in pairs:
            if left in by_label and right in by_label:
                return by_label[left], left, by_label[right], right
    raise ValueError("Could not find visually different endpoint labels in the test dataset.")


def render_panel(images: Any, labels: list[str], output: Path, *, scale: int) -> None:
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    tile_size = 28
    scaled = tile_size * scale
    gap = 6
    label_h = 30
    width = len(images) * scaled + (len(images) - 1) * gap
    height = label_h + scaled
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    nearest = Image.Resampling.NEAREST

    for index, image in enumerate(images):
        array = (image.squeeze(0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        tile = Image.fromarray(array, mode="L").convert("RGB")
        tile = tile.resize((scaled, scaled), resample=nearest)
        x = index * (scaled + gap)
        draw.text((x + 2, 6), labels[index], font=font, fill="#17130f")
        canvas.paste(tile, (x, label_h))

    canvas.save(output)


def main() -> int:
    args = parse_args()
    dataset_name = "fashion_mnist" if args.dataset == "fashion" else args.dataset
    checkpoint, payload = find_checkpoint(args.checkpoint)
    if checkpoint is None or payload is None:
        searched = ", ".join(str(path) for path in checkpoint_candidates()[:5]) or "no checkpoint files found"
        message = (
            "No trained VAE checkpoint was found.\n"
            "Train or provide one with: python train.py --model vae --dataset mnist --latent-dim 16\n"
            f"Searched latest candidates: {searched}"
        )
        print(message)
        write_placeholder(args.output, message)
        print(f"Wrote placeholder image: {args.output}")
        return 0

    try:
        import torch
        from train import VariationalAutoencoder
    except Exception as exc:
        message = f"Cannot generate interpolation because the VAE runtime imports failed: {exc}"
        print(message)
        write_placeholder(args.output, message)
        return 0

    latent_dim = latent_dim_from_payload(payload, args.latent_dim)
    model = VariationalAutoencoder(latent_dim=latent_dim)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()

    try:
        dataset, pairs, label_names = load_dataset(dataset_name, args.data_dir)
        left_image, left_label, right_image, right_label = choose_endpoints(dataset, pairs)
    except Exception as exc:
        message = f"Cannot generate interpolation because test data is unavailable: {exc}"
        print(message)
        write_placeholder(args.output, message)
        return 0

    endpoints = torch.stack([left_image, right_image], dim=0)
    with torch.no_grad():
        mu, _ = model.encode_features(endpoints)
        alphas = torch.linspace(0.0, 1.0, max(10, args.steps)).unsqueeze(1)
        latent_path = (1.0 - alphas) * mu[0].unsqueeze(0) + alphas * mu[1].unsqueeze(0)
        decoded = model.decode(latent_path).cpu()

    left_name = label_names[left_label]
    right_name = label_names[right_label]
    labels = ["start"] + ["" for _ in range(decoded.shape[0] - 2)] + ["end"]
    labels[0] = str(left_name)
    labels[-1] = str(right_name)
    render_panel(decoded, labels, args.output, scale=max(1, args.scale))
    print(f"Generated VAE interpolation from {left_name} to {right_name}: {args.output}")
    print(f"Checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
