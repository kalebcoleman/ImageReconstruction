from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ModuleNotFoundError:
    StructuralSimilarityIndexMeasure = None


def is_denoising(model_type: str) -> bool:
    return model_type == "dae"


def is_vae_model(model_type: str) -> bool:
    return model_type == "vae"


def compute_psnr(mse: float) -> float:
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def compute_batch_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> float:
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


def inject_noise(clean_images: torch.Tensor, noise_level: float) -> torch.Tensor:
    noisy_images = clean_images + noise_level * torch.randn_like(clean_images)
    return torch.clamp(noisy_images, 0.0, 1.0)


def compute_vae_loss(
    reconstructed_images: torch.Tensor,
    clean_images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    del criterion
    batch_size = clean_images.shape[0]
    recon_loss = nn.functional.binary_cross_entropy(
        reconstructed_images,
        clean_images,
        reduction="sum",
    ) / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + kl_loss, recon_loss


def run_forward_pass(
    model: nn.Module,
    inputs: torch.Tensor,
    clean_images: torch.Tensor,
    criterion: nn.Module,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_vae_model(model_type):
        reconstructed_images, mu, logvar = model(inputs)
        loss, _ = compute_vae_loss(reconstructed_images, clean_images, mu, logvar, criterion)
    elif model_type in ("ae", "dae"):
        reconstructed_images = model(inputs)
        loss = criterion(reconstructed_images, clean_images)
    else:
        raise NotImplementedError(f"Forward pass for {model_type} not implemented.")
    return loss, reconstructed_images


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
    *,
    model_type: str,
    noise_level: float,
) -> float:
    model.train()
    running_loss = 0.0

    for clean_images, _ in loader:
        clean_images = clean_images.to(device, non_blocking=True)
        inputs = inject_noise(clean_images, noise_level) if is_denoising(model_type) else clean_images
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_forward_pass(model, inputs, clean_images, criterion, model_type)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    model_type: str,
    noise_level: float,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for clean_images, _ in loader:
            clean_images = clean_images.to(device, non_blocking=True)
            inputs = inject_noise(clean_images, noise_level) if is_denoising(model_type) else clean_images
            loss, _ = run_forward_pass(model, inputs, clean_images, criterion, model_type)
            running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    model_type: str,
) -> dict[str, float]:
    model.eval()
    mse_total = 0.0
    ssim_total = 0.0
    num_batches = 0
    ssim_metric = None
    if StructuralSimilarityIndexMeasure is not None:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for clean_images, _ in loader:
            clean_images = clean_images.to(device, non_blocking=True)
            if is_vae_model(model_type):
                reconstructed_images = model.reconstruct(clean_images)
            elif model_type in ("ae", "dae"):
                reconstructed_images = model(clean_images)
            else:
                raise NotImplementedError(f"Metrics not supported for {model_type}.")

            mse_total += criterion(reconstructed_images, clean_images).item()
            if ssim_metric is not None:
                ssim_total += ssim_metric(reconstructed_images, clean_images).item()
                ssim_metric.reset()
            else:
                ssim_total += compute_batch_ssim(reconstructed_images, clean_images)
            num_batches += 1

    mse = mse_total / max(num_batches, 1)
    return {
        "mse": mse,
        "psnr": compute_psnr(mse),
        "ssim": ssim_total / max(num_batches, 1),
    }
