from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from autoencoders.training import inject_noise, is_denoising, is_vae_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_interpolation(image: torch.Tensor, interpolation: str | None = "nearest") -> str | None:
    if interpolation == "auto":
        return "nearest" if max(image.shape[-2:]) <= 32 else None
    return interpolation


def _plot_image_row(images: torch.Tensor, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().float().clamp(0.0, 1.0)
    figure, axes = plt.subplots(1, images.shape[0], figsize=(1.5 * images.shape[0], 2.0))
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, images):
        axis.imshow(image.squeeze(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_latent_space(model: nn.Module, loader: DataLoader, device: torch.device, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    latent_vectors: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for clean_images, labels in loader:
            clean_images = clean_images.to(device, non_blocking=True)
            latent_vectors.append(model.encode(clean_images).cpu().numpy())
            labels_list.append(labels.numpy())

    all_latent = np.concatenate(latent_vectors, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    reduced = PCA(n_components=2).fit_transform(all_latent)

    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=all_labels,
        cmap="tab10",
        s=10,
        alpha=0.75,
    )
    axis.set_title("Latent Space Representation")
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    figure.colorbar(scatter, ax=axis, label="Digit")
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def show_reconstructions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    *,
    model_type: str,
    noise_level: float,
    interpolation: str | None = "nearest",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    clean_images, _ = next(iter(loader))
    num_images = min(10, clean_images.shape[0])
    clean_images = clean_images[:num_images].to(device)

    inputs = clean_images
    noisy_images = None
    if is_denoising(model_type):
        noisy_images = inject_noise(clean_images, noise_level)
        inputs = noisy_images

    with torch.no_grad():
        reconstructed_images = model.reconstruct(inputs) if is_vae_model(model_type) else model(inputs)

    clean_images = clean_images.cpu()
    reconstructed_images = reconstructed_images.cpu()
    image_kwargs: dict[str, Any] = {"cmap": "gray"}
    resolved_interpolation = _resolve_interpolation(clean_images[0], interpolation=interpolation)
    if resolved_interpolation is not None:
        image_kwargs["interpolation"] = resolved_interpolation

    if noisy_images is not None:
        noisy_images = noisy_images.cpu()
        figure, axes = plt.subplots(3, num_images, figsize=(1.5 * num_images, 4), squeeze=False)
        rows = (clean_images, noisy_images, reconstructed_images)
        titles = ("Original", "Noisy", "Reconstructed")
    else:
        figure, axes = plt.subplots(2, num_images, figsize=(1.5 * num_images, 3), squeeze=False)
        rows = (clean_images, reconstructed_images)
        titles = ("Original", "Reconstructed")

    for row_idx, row_images in enumerate(rows):
        for col_idx in range(num_images):
            axes[row_idx, col_idx].imshow(row_images[col_idx].squeeze(), **image_kwargs)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_title(titles[row_idx])

    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_samples(
    model: nn.Module,
    latent_dim: int,
    device: torch.device,
    *,
    save_path: Path,
    num_samples: int,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        generated = model.decoder(z).view(-1, 1, 28, 28).cpu()
    _plot_image_row(generated, save_path, title="VAE Generated Samples")


def interpolate_images(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    save_path: Path,
    steps: int = 10,
) -> None:
    model.eval()
    image_batch: list[torch.Tensor] = []
    for clean_images, _ in loader:
        image_batch.append(clean_images)
        if sum(batch.shape[0] for batch in image_batch) >= 2:
            break

    if not image_batch:
        raise ValueError("Loader does not contain any images.")

    source_images = torch.cat(image_batch, dim=0)[:2].to(device)
    if source_images.shape[0] < 2:
        raise ValueError("Interpolation requires at least two images.")

    with torch.no_grad():
        mu, _ = model.encode_features(source_images)
        z1, z2 = mu[0], mu[1]
        alphas = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)
        latent_path = (1.0 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)
        interpolated = model.decoder(latent_path).view(-1, 1, 28, 28).cpu()

    _plot_image_row(interpolated, save_path, title="VAE Latent Interpolation")
