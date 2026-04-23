from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ModuleNotFoundError:  # pragma: no cover - fallback used in lean envs.
    StructuralSimilarityIndexMeasure = None

from diffusion.runtime import autocast_context
from diffusion.ema import update_ema_model
from diffusion.scheduler import (
    DiffusionSchedule,
    get_diffusion_target,
    predict_x0_from_model_output,
    q_sample,
)


LOGGER = logging.getLogger(__name__)


def _move_batch_to_device(
    batch: tuple[torch.Tensor, ...] | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """Move every tensor in a batch tuple onto the active compute device."""

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return tuple(
        item.to(device, non_blocking=True) if torch.is_tensor(item) else item
        for item in batch
    )


def _unpack_diffusion_batch(batch: tuple[torch.Tensor, ...] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return images plus an optional label tensor from a diffusion batch."""

    if torch.is_tensor(batch):
        return batch, None
    if len(batch) == 1:
        return batch[0], None
    return batch[0], batch[1]


def _sample_timesteps(batch_size: int, schedule: DiffusionSchedule, device: torch.device) -> torch.Tensor:
    """Draw one random diffusion step for each image in the batch."""
    return torch.randint(
        low=0,
        high=schedule.num_timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )


def _validate_diffusion_shapes(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
) -> None:
    """Ensure the denoiser returns the same image shape as the diffusion target."""

    if predicted_noise.shape != target_noise.shape:
        raise ValueError(
            "Diffusion model output shape must match the training target. "
            f"Expected {tuple(target_noise.shape)}, got {tuple(predicted_noise.shape)}."
        )


def _compute_psnr(mse: float) -> float:
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def _diffusion_to_display_range(images: torch.Tensor) -> torch.Tensor:
    """Map diffusion tensors from [-1, 1] to [0, 1] for image-space metrics."""

    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


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


def _should_log_progress(batch_idx: int, total_batches: int, progress_interval: int | None) -> bool:
    """Emit progress at a fixed interval and on the final batch."""
    if progress_interval is None or progress_interval <= 0:
        return False
    return batch_idx % progress_interval == 0 or batch_idx == total_batches


def train_diffusion_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: DiffusionSchedule,
    device: torch.device,
    prediction_type: str = "eps",
    ema_model: torch.nn.Module | None = None,
    ema_decay: float = 0.0,
    amp_dtype: str = "none",
    grad_clip_norm: float | None = None,
    grad_scaler: torch.cuda.amp.GradScaler | None = None,
    progress_label: str | None = None,
    progress_interval: int | None = None,
) -> float:
    """Train for one epoch with the standard DDPM noise-prediction objective.

    We corrupt each clean image x0 into x_t, ask the model to predict the exact
    Gaussian noise that was used, and optimize an MSE loss. Predicting noise is
    convenient because the target distribution is simple and the clean image can
    be reconstructed from x_t plus the predicted epsilon.
    """

    model.train()
    running_loss = 0.0
    scaler_enabled = grad_scaler is not None and grad_scaler.is_enabled()

    total_batches = max(len(loader), 1)

    for batch_idx, batch in enumerate(loader, start=1):
        images, labels = _unpack_diffusion_batch(_move_batch_to_device(batch, device))
        timesteps = _sample_timesteps(images.shape[0], scheduler, device)
        noise = torch.randn(images.shape, device=device, dtype=images.dtype)
        noisy_images = q_sample(images, timesteps, noise, scheduler)
        target = get_diffusion_target(images, noise, timesteps, scheduler, prediction_type)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_dtype, device):
            model_output = model(noisy_images, timesteps, labels)
            _validate_diffusion_shapes(model_output, target)
            loss = F.mse_loss(model_output, target)

        if scaler_enabled:
            grad_scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        if ema_model is not None:
            update_ema_model(ema_model, model, ema_decay)

        running_loss += loss.item()

        if progress_label and _should_log_progress(batch_idx, total_batches, progress_interval):
            average_loss = running_loss / batch_idx
            LOGGER.info(
                f"{progress_label} | Train Batch {batch_idx}/{total_batches} | "
                f"Avg Loss={average_loss:.6f}"
            )

    return running_loss / total_batches


@torch.no_grad()
def eval_diffusion_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler: DiffusionSchedule,
    device: torch.device,
    prediction_type: str = "eps",
    amp_dtype: str = "none",
    progress_label: str | None = None,
    progress_interval: int | None = None,
    eval_split_name: str = "Val",
) -> float:
    """Evaluation loss for diffusion uses the same noise-prediction objective."""

    model.eval()
    running_loss = 0.0

    total_batches = max(len(loader), 1)

    for batch_idx, batch in enumerate(loader, start=1):
        images, labels = _unpack_diffusion_batch(_move_batch_to_device(batch, device))
        timesteps = _sample_timesteps(images.shape[0], scheduler, device)
        noise = torch.randn(images.shape, device=device, dtype=images.dtype)
        noisy_images = q_sample(images, timesteps, noise, scheduler)
        target = get_diffusion_target(images, noise, timesteps, scheduler, prediction_type)
        with autocast_context(amp_dtype, device):
            model_output = model(noisy_images, timesteps, labels)
            _validate_diffusion_shapes(model_output, target)
            batch_loss = F.mse_loss(model_output, target)
        running_loss += batch_loss.item()

        if progress_label and _should_log_progress(batch_idx, total_batches, progress_interval):
            average_loss = running_loss / batch_idx
            LOGGER.info(
                f"{progress_label} | {eval_split_name} Batch {batch_idx}/{total_batches} | "
                f"Avg Loss={average_loss:.6f}"
            )

    return running_loss / total_batches


@torch.no_grad()
def evaluate_diffusion_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    scheduler: DiffusionSchedule,
    device: torch.device,
    prediction_type: str = "eps",
    amp_dtype: str = "none",
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

    for batch in loader:
        images, labels = _unpack_diffusion_batch(_move_batch_to_device(batch, device))
        timesteps = _sample_timesteps(images.shape[0], scheduler, device)
        noise = torch.randn(images.shape, device=device, dtype=images.dtype)
        noisy_images = q_sample(images, timesteps, noise, scheduler)
        with autocast_context(amp_dtype, device):
            model_output = model(noisy_images, timesteps, labels)
            _validate_diffusion_shapes(model_output, noise)
        reconstructed = predict_x0_from_model_output(
            noisy_images,
            timesteps,
            model_output,
            scheduler,
            prediction_type,
        ).clamp(-1.0, 1.0)

        reconstructed_for_metrics = _diffusion_to_display_range(reconstructed)
        images_for_metrics = _diffusion_to_display_range(images)

        batch_mse = F.mse_loss(reconstructed_for_metrics, images_for_metrics).item()
        if ssim_metric is not None:
            batch_ssim = ssim_metric(reconstructed_for_metrics, images_for_metrics).item()
            ssim_metric.reset()
        else:
            batch_ssim = _compute_batch_ssim(reconstructed_for_metrics, images_for_metrics)

        mse_total += batch_mse
        ssim_total += batch_ssim
        num_batches += 1

    mse = mse_total / max(num_batches, 1)
    return {
        "mse": mse,
        "psnr": _compute_psnr(mse),
        "ssim": ssim_total / max(num_batches, 1),
    }
