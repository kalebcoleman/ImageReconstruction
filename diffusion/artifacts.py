from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion.sampling import sample_images
from diffusion.scheduler import predict_x0_from_model_output, q_sample

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_IMAGE_INTERPOLATION = "nearest"
AUTO_IMAGE_INTERPOLATION = "auto"
LOW_RES_PRESENTATION_MAX_SIZE = 32


def prepare_display_images(images: torch.Tensor, *, rescale: bool = False) -> torch.Tensor:
    display_images = images.detach().cpu().float()
    if rescale:
        flat = display_images.reshape(display_images.shape[0], -1)
        mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
        display_images = (display_images - mins) / (maxs - mins).clamp_min(1e-8)
    return display_images.clamp(0.0, 1.0)


def diffusion_to_display_range(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def image_for_plot(image: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    if image.ndim != 3:
        raise ValueError(f"Expected a CHW image tensor, got shape {tuple(image.shape)}.")

    if image.shape[0] == 1:
        return image.squeeze(0).numpy(), {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
    if image.shape[0] >= 3:
        return image[:3].permute(1, 2, 0).clamp(0.0, 1.0).numpy(), {}
    raise ValueError(f"Unsupported channel count for plotting: {image.shape[0]}.")


def resolve_image_interpolation(
    image: torch.Tensor,
    *,
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
) -> str | None:
    if interpolation != AUTO_IMAGE_INTERPOLATION:
        return interpolation
    if max(image.shape[-2:]) <= LOW_RES_PRESENTATION_MAX_SIZE:
        return DEFAULT_IMAGE_INTERPOLATION
    return None


def render_image(
    axis: Any,
    image: torch.Tensor,
    *,
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
) -> None:
    plot_tensor = image.detach().cpu().float()
    plot_image, render_kwargs = image_for_plot(plot_tensor)
    resolved_interpolation = resolve_image_interpolation(plot_tensor, interpolation=interpolation)
    if resolved_interpolation is not None:
        render_kwargs = {**render_kwargs, "interpolation": resolved_interpolation}
    axis.imshow(plot_image, **render_kwargs)


def plot_image_grid(
    images: torch.Tensor,
    save_path: Path,
    title: str,
    *,
    num_cols: int = 5,
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images = prepare_display_images(images)
    num_images = images.shape[0]
    num_cols = max(1, min(num_cols, num_images))
    num_rows = math.ceil(num_images / num_cols)
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(1.8 * num_cols, 1.8 * num_rows))
    axes_array = np.atleast_1d(axes).reshape(num_rows, num_cols)

    for image_idx in range(num_rows * num_cols):
        axis = axes_array.flat[image_idx]
        axis.axis("off")
        if image_idx < num_images:
            render_image(axis, images[image_idx], interpolation=interpolation)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_native_image_grid(
    images: torch.Tensor,
    save_path: Path,
    *,
    num_cols: int | None = None,
    padding: int = 0,
    scale: int = 1,
) -> None:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if scale < 1:
        raise ValueError("scale must be at least 1")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    display_images = prepare_display_images(images).detach().cpu().float().clamp(0.0, 1.0)
    num_images, channels, height, width = display_images.shape
    if num_images < 1:
        raise ValueError("images must contain at least one image")

    resolved_cols = num_cols or math.ceil(math.sqrt(num_images))
    resolved_cols = max(1, min(resolved_cols, num_images))
    num_rows = math.ceil(num_images / resolved_cols)
    sheet = Image.new(
        "RGB",
        (
            resolved_cols * width + (resolved_cols - 1) * padding,
            num_rows * height + (num_rows - 1) * padding,
        ),
        "white",
    )

    for image_idx, image in enumerate(display_images):
        rgb_image = image.repeat(3, 1, 1) if channels == 1 else image[:3]
        array = (rgb_image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        tile = Image.fromarray(array, mode="RGB")
        x = (image_idx % resolved_cols) * (width + padding)
        y = (image_idx // resolved_cols) * (height + padding)
        sheet.paste(tile, (x, y))

    if scale > 1:
        nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        sheet = sheet.resize((sheet.width * scale, sheet.height * scale), resample=nearest)
    sheet.save(save_path)


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
    *,
    title: str | None = None,
    y_label: str = "Loss",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", linewidth=2.0, markersize=5, label="Train Loss")
    axis.plot(epochs, val_losses, marker="s", linewidth=2.0, markersize=5, label="Validation Loss")
    if title is not None:
        axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_diffusion_snapshots(
    model: nn.Module,
    scheduler: object,
    device: torch.device,
    *,
    dataset_name: str,
    image_shape: tuple[int, int, int],
    base_channels: int,
    save_path: Path,
    num_samples: int,
    sample_labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    prediction_type: str = "eps",
    sampler_name: str = "ddpm",
    sampling_steps: int | None = None,
    ddim_eta: float = 0.0,
    amp_dtype: str = "none",
    num_snapshots: int = 9,
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _, intermediate_images, intermediate_steps = sample_images(
        model,
        scheduler,
        device,
        num_samples=num_samples,
        image_shape=image_shape,
        labels=sample_labels,
        guidance_scale=guidance_scale,
        prediction_type=prediction_type,
        sampler_name=sampler_name,
        sampling_steps=sampling_steps,
        ddim_eta=ddim_eta,
        amp_dtype=amp_dtype,
        return_intermediate=True,
        num_snapshots=num_snapshots,
    )

    figure, axes = plt.subplots(
        len(intermediate_images),
        num_samples,
        figsize=(1.55 * num_samples, 1.55 * len(intermediate_images)),
        squeeze=False,
    )

    for row_idx, (images, step_num) in enumerate(zip(intermediate_images, intermediate_steps)):
        display_images = prepare_display_images(diffusion_to_display_range(images), rescale=True)
        for col_idx in range(num_samples):
            axis = axes[row_idx, col_idx]
            render_image(axis, display_images[col_idx], interpolation=interpolation)
            axis.axis("off")
            if col_idx == 0:
                axis.set_ylabel(f"t={step_num}", rotation=0, labelpad=24, va="center", fontsize=9, fontweight="bold")

    figure.suptitle(
        f"Reverse Diffusion Process (Noise -> Image)\n"
        f"{dataset_name.upper()} | Base Width = {base_channels} | Sample Shape = {image_shape} | {prediction_type}/{sampler_name}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_diffusion_reconstructions(
    model: nn.Module,
    scheduler: object,
    loader: DataLoader,
    device: torch.device,
    *,
    dataset_name: str,
    base_channels: int,
    prediction_type: str,
    save_path: Path,
    num_images: int = 8,
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    clean_images, labels = next(iter(loader))
    num_images = min(num_images, clean_images.shape[0])
    clean_images = clean_images[:num_images].to(device)
    labels = labels[:num_images].to(device)

    preview_step = max(1, scheduler.num_timesteps // 2)
    timesteps = torch.full((num_images,), preview_step, device=device, dtype=torch.long)
    noise = torch.randn_like(clean_images)
    noisy_images = q_sample(clean_images, timesteps, noise, scheduler)

    with torch.no_grad():
        model_output = model(noisy_images, timesteps, labels)
        reconstructed_images = predict_x0_from_model_output(
            noisy_images,
            timesteps,
            model_output,
            scheduler,
            prediction_type,
        ).clamp(-1.0, 1.0)

    clean_images = diffusion_to_display_range(clean_images)
    noisy_images = diffusion_to_display_range(noisy_images)
    reconstructed_images = diffusion_to_display_range(reconstructed_images)

    figure, axes = plt.subplots(3, num_images, figsize=(1.5 * num_images, 4.2), squeeze=False)
    row_titles = ("Original", f"Noisy (t={preview_step})", "Predicted x0")
    image_rows = (clean_images, noisy_images, reconstructed_images)

    for row_idx, (row_title, row_images) in enumerate(zip(row_titles, image_rows)):
        for col_idx in range(num_images):
            axis = axes[row_idx, col_idx]
            render_image(axis, row_images[col_idx], interpolation=interpolation)
            axis.axis("off")
            if col_idx == 0:
                axis.set_title(row_title)

    figure.suptitle(
        f"Diffusion Reconstruction Preview ({dataset_name.upper()}, Base Width={base_channels}, {prediction_type})",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
