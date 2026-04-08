from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ModuleNotFoundError:  # pragma: no cover - fallback used in lean envs.
    StructuralSimilarityIndexMeasure = None

from diffusion.scheduler import DiffusionSchedule, predict_x0_from_noise, q_sample


def _sample_timesteps(batch_size: int, schedule: DiffusionSchedule, device: torch.device) -> torch.Tensor:
    """Draw one random diffusion step for each image in the batch."""
    return torch.randint(
        low=0,
        high=schedule.num_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )


def _compute_psnr(mse: float) -> float:
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def _compute_batch_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute a lightweight global SSIM approximation in pure PyTorch."""

    c1 = 0.01**2
    c2 = 0.03**2

    reduce_dims = tuple(range(1, predictions.ndim))
    mu_x = predictions.mean(dim=reduce_dims)
    mu_y = targets.mean(dim=reduce_dims)

    centered_x = predictions - mu_x.view(-1, 1, 1, 1)
    centered_y = targets - mu_y.view(-1, 1, 1, 1)
    sigma_x = centered_x.square().mean(dim=reduce_dims)
    sigma_y = centered_y.square().mean(dim=reduce_dims)
    sigma_xy = (centered_x * centered_y).mean(dim=reduce_dims)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp(min=1e-12)).mean().item()


def train_diffusion_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: DiffusionSchedule,
    device: torch.device,
) -> float:
    """Train for one epoch with the standard DDPM noise-prediction objective.

    We corrupt each clean image x0 into x_t, ask the model to predict the exact
    Gaussian noise that was used, and optimize an MSE loss. Predicting noise is
    convenient because the target distribution is simple and the clean image can
    be reconstructed from x_t plus the predicted epsilon.
    """

    model.train()
    running_loss = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        timesteps = _sample_timesteps(images.shape[0], scheduler, device)
        noise = torch.randn_like(images)
        noisy_images = q_sample(images, timesteps, noise, scheduler)

        optimizer.zero_grad(set_to_none=True)
        predicted_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def eval_diffusion_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler: DiffusionSchedule,
    device: torch.device,
) -> float:
    """Validation loss for diffusion uses the same noise-prediction objective."""

    model.eval()
    running_loss = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        timesteps = _sample_timesteps(images.shape[0], scheduler, device)
        noise = torch.randn_like(images)
        noisy_images = q_sample(images, timesteps, noise, scheduler)
        predicted_noise = model(noisy_images, timesteps)
        running_loss += F.mse_loss(predicted_noise, noise).item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_diffusion_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler: DiffusionSchedule,
    device: torch.device,
) -> dict[str, float]:
    """Estimate reconstruction-style metrics from denoising predictions.

    This keeps the evaluation interface comparable to AE/DAE/VAE pipelines:
    we corrupt real images at random timesteps, predict the noise, convert that
    prediction back into an x0 estimate, and then score the recovered image.
    """

    model.eval()
    ssim_metric = None
    if StructuralSimilarityIndexMeasure is not None:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse_total = 0.0
    ssim_total = 0.0
    num_batches = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        timesteps = _sample_timesteps(images.shape[0], scheduler, device)
        noise = torch.randn_like(images)
        noisy_images = q_sample(images, timesteps, noise, scheduler)
        predicted_noise = model(noisy_images, timesteps)
        reconstructed = predict_x0_from_noise(
            noisy_images,
            timesteps,
            predicted_noise,
            scheduler,
        ).clamp(0.0, 1.0)

        batch_mse = F.mse_loss(reconstructed, images).item()
        if ssim_metric is not None:
            batch_ssim = ssim_metric(reconstructed, images).item()
            ssim_metric.reset()
        else:
            batch_ssim = _compute_batch_ssim(reconstructed, images)

        mse_total += batch_mse
        ssim_total += batch_ssim
        num_batches += 1

    mse = mse_total / max(num_batches, 1)
    return {
        "mse": mse,
        "psnr": _compute_psnr(mse),
        "ssim": ssim_total / max(num_batches, 1),
    }
