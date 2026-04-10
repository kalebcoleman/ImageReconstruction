from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ModuleNotFoundError:
    StructuralSimilarityIndexMeasure = None
from torchvision import datasets, transforms

from diffusion.model import DiffusionUNet
from diffusion.scheduler import get_noise_schedule
from diffusion.training import (
    train_diffusion_epoch,
    eval_diffusion_epoch,
    evaluate_diffusion_metrics,
    _compute_batch_ssim,
)
from diffusion.sampling import sample_images

# Configure matplotlib for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIG & CONSTANTS
# ==============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for the training pipeline.
    Grouping parameters here makes it easy to track and change experiment settings.
    """
    data_dir: Path = Path("./data")          # Directory to download/load dataset
    output_dir: Path = Path("./outputs")     # Directory to save all artifacts and results
    batch_size: int = 8                      # Number of samples per batch
    epochs: int = 1                          # Number of epochs to train
    learning_rate: float = 1e-3              # Learning rate for the optimizer
    n_splits: int = 2                        # K-Fold Cross Validation splits
    random_seed: int = 42                    # Seed for reproducibility
    latent_dims: tuple[int, ...] = (8,)      # Different latent dimensions to evaluate
    num_workers: int = 0                     # Number of workers for DataLoader
    diffusion_timesteps: int = 10            # Number of timesteps for diffusion models
    diffusion_log_interval: int = 250        # Batch interval for diffusion progress logs
    diffusion_val_fraction: float = 0.1      # Deprecated: diffusion now uses the standard train/test split

CONFIG = ExperimentConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device
CRITERION = nn.MSELoss()                     # Standard loss for AE and DAE

# Cache datasets in memory to avoid multiple loading operations
DATASET_CACHE: dict[tuple[str, bool], datasets.VisionDataset] = {}

# Constants for running experiments
RUN_DATASETS = ["mnist"]                     # Add more datasets if needed
RUN_MODELS = ["ae", "dae", "vae", "diffusion"]
DAE_NOISE_LEVEL = 0.2                        # Default noise standard deviation for DAE


# ==============================================================================
# 2. MODELS (Autoencoder, Variational Autoencoder)
# ==============================================================================

class FullyConnectedAutoencoder(nn.Module):
    """
    Standard Fully Connected Autoencoder.
    Compresses 28x28 images into a latent dimension and reconstructs them.
    Used for both standard AE and Denoising AE (DAE).
    """
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        # Reshape flat output back into 1-channel image (e.g. 1x28x28 for MNIST)
        return decoded.view(-1, 1, 28, 28)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        return self.decode(latent)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE).
    Instead of projecting to a single latent point, it projects to a probability distribution,
    which provides a more continuous and structured latent space.
    """
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        hidden_dim = 400
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
        )
        # VAE has separate linear layers for the mean (mu) and log-variance (logvar) of the distribution
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes input to mu and logvar."""
        hidden = self.encoder(self.flatten(x))
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The Reparameterization Trick.
        Allows backpropagation through random sampling by computing: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns just the mean of the latent distribution for downstream tasks."""
        mu, _ = self.encode_features(x)
        return mu

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        return decoded.view(-1, 1, 28, 28)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning just the reconstructed output."""
        mu, _ = self.encode_features(x)
        return self.decode(mu)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_features(inputs)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent)
        return reconstructed, mu, logvar


# ==============================================================================
# 3. HELPER FUNCTIONS (Routing, Utilities)
# ==============================================================================

def set_seed(seed: int) -> None:
    """Sets seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_denoising(model_type: str) -> bool:
    """Explicitly check if the model type corresponds to a denoising task."""
    return model_type == "dae"

def is_vae_model(model_type: str) -> bool:
    """Explicitly check if the model type corresponds to a VAE."""
    return model_type == "vae"

def is_diffusion(model_type: str) -> bool:
    """Check if model type is diffusion."""
    return model_type == "diffusion"


def ensure_diffusion_output_dirs(output_root: Path) -> dict[str, Path]:
    """Create and return the standard diffusion artifact directories."""
    diffusion_root = output_root / "diffusion"
    paths = {
        "root": diffusion_root,
        "loss_curves": diffusion_root / "loss_curves",
        "samples": diffusion_root / "samples",
        "snapshots": diffusion_root / "snapshots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def format_experiment_name(
    config: dict[str, str | int],
    artifact: str,
    *,
    include_epochs: bool = False,
) -> str:
    """Build consistent diffusion artifact filenames."""
    parts = [
        str(config["dataset"]).lower(),
        artifact,
        f"bc{config['base_channels']}",
        f"t{config['timesteps']}",
    ]
    if include_epochs:
        parts.append(f"e{config['epochs']}")
    return "_".join(parts)


def build_diffusion_artifact_config(dataset_name: str, base_channels: int) -> dict[str, str | int]:
    """Centralize diffusion naming inputs so plots and logs stay aligned."""
    return {
        "dataset": dataset_name,
        "base_channels": base_channels,
        "timesteps": CONFIG.diffusion_timesteps,
        "epochs": CONFIG.epochs,
    }


def print_diffusion_experiment_header(dataset_name: str, base_channels: int) -> None:
    """Emit a clean experiment banner for diffusion runs."""
    print("-" * 40)
    print("Running Diffusion Model")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Base Channels: {base_channels}")
    print(f"Timesteps: {CONFIG.diffusion_timesteps}")
    print(f"Epochs: {CONFIG.epochs}")
    print("-" * 40)


def prepare_display_images(images: torch.Tensor, *, rescale: bool = False) -> torch.Tensor:
    """Normalize tensors for report-friendly visualization without touching model behavior."""
    display_images = images.detach().cpu().float()
    if rescale:
        flat = display_images.reshape(display_images.shape[0], -1)
        mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
        display_images = (display_images - mins) / (maxs - mins).clamp_min(1e-8)
    return display_images.clamp(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the default experiment sweep."""
    parser = argparse.ArgumentParser(description="Train image reconstruction models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=RUN_MODELS,
        default=RUN_MODELS,
        help="Subset of models to train. Defaults to the full configured sweep.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=RUN_DATASETS,
        default=RUN_DATASETS,
        help="Subset of datasets to train on. Defaults to the full configured sweep.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the DataLoader batch size.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="Override the number of K-Fold splits. Must be at least 2.",
    )
    parser.add_argument(
        "--base-channels",
        nargs="+",
        type=int,
        dest="base_channels",
        help="Override diffusion UNet base channel widths.",
    )
    parser.add_argument(
        "--latent-dims",
        nargs="+",
        type=int,
        dest="base_channels",
        help="Deprecated alias for --base-channels. Values are mapped internally to diffusion base channels.",
    )
    parser.add_argument(
        "--diffusion-timesteps",
        type=int,
        help="Override the number of diffusion timesteps used for training and sampling.",
    )
    parser.add_argument(
        "--diffusion-log-interval",
        type=int,
        help="Override the batch interval for diffusion progress logging.",
    )
    parser.add_argument(
        "--diffusion-val-fraction",
        type=float,
        help="Override the holdout validation fraction used for diffusion.",
    )
    args = parser.parse_args()
    if args.n_splits is not None and args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    if args.diffusion_val_fraction is not None and not 0.0 < args.diffusion_val_fraction < 1.0:
        parser.error("--diffusion-val-fraction must be between 0 and 1")
    return args

def get_dataset(dataset_name: str = "mnist", train: bool = True) -> datasets.VisionDataset:
    """
    Fetches the requested dataset, leveraging caching to avoid redundant downloads/loading.
    """
    dataset_key = dataset_name.lower()
    dataset_classes: dict[str, type[datasets.VisionDataset]] = {
        "mnist": datasets.MNIST,
        "fashion": datasets.FashionMNIST,
        # Future datasets can be added here
    }
    if dataset_key not in dataset_classes:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    cache_key = (dataset_key, train)
    if cache_key not in DATASET_CACHE:
        DATASET_CACHE[cache_key] = dataset_classes[dataset_key](
            root=CONFIG.data_dir,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
    return DATASET_CACHE[cache_key]

def create_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    """Centralized dataloader configuration."""
    return DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_experiment_splits(dataset_size: int, model_type: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return CV folds for reconstruction models."""
    splitter = KFold(
        n_splits=CONFIG.n_splits,
        shuffle=True,
        random_state=CONFIG.random_seed,
    )
    return [
        (fold_idx, train_indices, val_indices)
        for fold_idx, (train_indices, val_indices) in enumerate(splitter.split(range(dataset_size)), start=1)
    ]

def compute_psnr(mse: float) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    Higher is generally better, meaning lower reconstruction error.
    """
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def is_finite_metric(value: float) -> bool:
    """Guard plotting/reporting paths that should skip undefined metrics."""
    return math.isfinite(value)


def format_metric(value: float, precision: int = 4) -> str:
    """Render undefined metrics as N/A so diffusion reporting stays explicit."""
    if not is_finite_metric(value):
        return "N/A"
    return f"{value:.{precision}f}"


def format_capacity_label(model_type: str, capacity: int) -> str:
    """Render model capacity using the right term for each architecture."""
    if is_diffusion(model_type):
        return f"Base Channels={capacity}"
    return f"Latent={capacity}"


def build_diffusion_metrics(noise_prediction_mse: float) -> dict[str, float]:
    """Diffusion validation tracks noise-prediction MSE, not image reconstruction metrics."""
    return {
        "mse": noise_prediction_mse,
        "psnr": float("nan"),
        "ssim": float("nan"),
    }


def mean_metric(values: list[float]) -> float:
    """Average only defined values so diffusion can skip unsupported metrics cleanly."""
    finite_values = [value for value in values if is_finite_metric(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def inject_noise(clean_images: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Injects random Gaussian noise into the images to simulate noisy inputs
    for the Denoising Autoencoder (DAE).
    """
    noise = noise_level * torch.randn_like(clean_images)
    # Clamp to ensure image pixels remain in the valid range [0, 1]
    noisy_images = torch.clamp(clean_images + noise, 0.0, 1.0)
    return noisy_images


def compute_vae_loss(
    reconstructed_images: torch.Tensor,
    clean_images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Variational Autoencoder loss consists of two parts:
    
    1. Reconstruction Loss (Binary Cross Entropy - BCE):
       - Measures how well the decoder recreates the original image from the latent vector.
       - BCE is used here interpreting pixels as probabilities (since they are in [0,1] for MNIST).
    
    2. KL Divergence:
       - Measures how much the learned latent distribution diverges from a standard normal distribution.
       - It acts as a regularizer, forcing the latent space to be continuous and well-structured,
         which enables generating new samples by sampling from N(0, 1).
         
    Both are needed to ensure the latent space is well behaved while retaining fidelity.
    """
    del criterion # Not used for VAE, we use BCE directly
    batch_size = clean_images.shape[0]
    
    recon_loss = nn.functional.binary_cross_entropy(
        reconstructed_images, clean_images, reduction="sum"
    ) / batch_size
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss


def run_forward_pass(
    model: nn.Module,
    inputs: torch.Tensor,
    clean_images: torch.Tensor,
    criterion: nn.Module,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shared forward pass for both training and evaluation loops.
    Handles routing based on the extensible 'model_type'.
    """
    if is_vae_model(model_type):
        reconstructed_images, mu, logvar = model(inputs)
        loss, _ = compute_vae_loss(reconstructed_images, clean_images, mu, logvar, criterion)
    elif model_type in ("ae", "dae"):
        reconstructed_images = model(inputs)
        loss = criterion(reconstructed_images, clean_images)
    else:
        # Easy to extend for future models (e.g., "diffusion")
        raise NotImplementedError(f"Forward pass for {model_type} not implemented yet.")
        
    return loss, reconstructed_images


# ==============================================================================
# 4. TRAINING & EVALUATION
# ==============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "ae",
    noise_level: float = 0.2,
) -> float:
    """
    Runs a single training epoch across all batches in the data loader.
    """
    model.train()
    running_loss = 0.0

    for clean_images, _ in loader:
        clean_images = clean_images.to(device, non_blocking=True)
        
        # Define model inputs (clean or noisy)
        inputs = clean_images
        if is_denoising(model_type):
            inputs = inject_noise(clean_images, noise_level)

        optimizer.zero_grad(set_to_none=True)
        
        # Shared forward logic
        loss, _ = run_forward_pass(model, inputs, clean_images, criterion, model_type)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "ae",
    noise_level: float = 0.2,
) -> float:
    """
    Evaluates the model on the validation dataset (no gradients).
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for clean_images, _ in loader:
            clean_images = clean_images.to(device, non_blocking=True)
            
            inputs = clean_images
            if is_denoising(model_type):
                inputs = inject_noise(clean_images, noise_level)
            
            loss, _ = run_forward_pass(model, inputs, clean_images, criterion, model_type)
            running_loss += loss.item()

    return running_loss / len(loader)


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "ae",
) -> dict[str, float]:
    """
    Computes rigorous evaluation metrics (MSE, PSNR, SSIM) for final validation.
    """
    model.eval()
    mse_total = 0.0
    ssim_total = 0.0
    num_batches = 0
    # Structural Similarity Index Measure: compares image structural differences
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
                raise NotImplementedError(f"Metrics not supported for {model_type}")
                
            mse_total += criterion(reconstructed_images, clean_images).item()
            if ssim_metric is not None:
                ssim_total += ssim_metric(reconstructed_images, clean_images).item()
                ssim_metric.reset()
            else:
                ssim_total += _compute_batch_ssim(reconstructed_images, clean_images)
            num_batches += 1

    mse = mse_total / num_batches
    return {
        "mse": mse,
        "psnr": compute_psnr(mse),
        "ssim": ssim_total / num_batches,
    }

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================

def show_reconstructions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    model_type: str = "ae",
    noise_level: float = 0.2,
) -> None:
    """Visualize original vs (optionally noisy) vs reconstructed images."""
    model.eval()
    clean_images, _ = next(iter(loader))
    num_images = min(10, clean_images.shape[0])
    clean_images = clean_images[:num_images].to(device)

    inputs = clean_images
    if is_denoising(model_type):
        noisy_images = inject_noise(clean_images, noise_level)
        inputs = noisy_images

    with torch.no_grad():
        if is_vae_model(model_type):
            reconstructed_images = model.reconstruct(inputs)
        else:
            reconstructed_images = model(inputs)

    clean_images = clean_images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    if is_denoising(model_type):
        noisy_images = noisy_images.cpu()
        figure, axes = plt.subplots(3, num_images, figsize=(1.5 * num_images, 4), squeeze=False)
        for i in range(num_images):
            axes[0, i].imshow(clean_images[i].squeeze(), cmap="gray")
            axes[1, i].imshow(noisy_images[i].squeeze(), cmap="gray")
            axes[2, i].imshow(reconstructed_images[i].squeeze(), cmap="gray")
            for j in range(3):
                axes[j, i].axis("off")
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Noisy")
        axes[2, 0].set_title("Reconstructed")
    else:
        figure, axes = plt.subplots(2, num_images, figsize=(1.5 * num_images, 3), squeeze=False)
        for i in range(num_images):
            axes[0, i].imshow(clean_images[i].squeeze(), cmap="gray")
            axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Reconstructed")

    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def plot_image_row(images: torch.Tensor, save_path: Path, title: str) -> None:
    """Helper method to plot a simple 1D row of images."""
    num_images = images.shape[0]
    figure, axes = plt.subplots(1, num_images, figsize=(1.5 * num_images, 2.0))
    if num_images == 1:
        axes = [axes]
    for axis, image in zip(axes, images):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_image_grid(
    images: torch.Tensor,
    save_path: Path,
    title: str,
    num_cols: int = 5,
) -> None:
    """Save a compact grid of generated images for unconditional samplers."""
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
            axis.imshow(images[image_idx].squeeze(), cmap="gray")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_samples(
    model: nn.Module,
    latent_dim: int,
    device: torch.device,
    num_samples: int = 10,
    save_path: Path | None = None,
) -> None:
    """Generates new images for generative models (like VAE) from pure noise."""
    if save_path is None:
        raise ValueError("generate_samples requires save_path.")

    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        generated = model.decoder(z).view(-1, 1, 28, 28).cpu()

    plot_image_row(generated, save_path, title="VAE Generated Samples")

def interpolate_images(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int = 10,
    save_path: Path | None = None,
) -> None:
    """Interpolates between two images smoothly in the model's latent space."""
    if save_path is None:
        raise ValueError("interpolate_images requires save_path.")

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

    plot_image_row(interpolated, save_path, title="VAE Latent Interpolation")


def plot_diffusion_snapshots(
    model: nn.Module,
    scheduler: object,
    device: torch.device,
    dataset_name: str,
    base_channels: int,
    save_path: Path,
    num_samples: int = 8,
    num_snapshots: int = 9,
) -> None:
    """Generates a progression visual of the diffusion reverse process."""
    _, intermediate_images, intermediate_steps = sample_images(
        model,
        scheduler,
        device,
        num_samples=num_samples,
        return_intermediate=True,
        num_snapshots=num_snapshots,
    )

    num_snapshots = len(intermediate_images)
    figure, axes = plt.subplots(
        num_snapshots,
        num_samples,
        figsize=(1.55 * num_samples, 1.55 * num_snapshots),
        squeeze=False,
    )

    for row_idx, (images, step_num) in enumerate(zip(intermediate_images, intermediate_steps)):
        display_images = prepare_display_images(images, rescale=True)
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            ax.imshow(display_images[col_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(
                    f"t={step_num}",
                    rotation=0,
                    labelpad=24,
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    figure.suptitle(
        f"Reverse Diffusion Process (Noise -> Image)\n"
        f"{dataset_name.upper()} | Base Channels = {base_channels}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_latent_space(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> None:
    """Reduces the latent space into 2D via PCA to visualize clustering by class labels."""
    model.eval()
    latent_vectors: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    with torch.no_grad():
        for clean_images, labels in loader:
            clean_images = clean_images.to(device, non_blocking=True)
            latent = model.encode(clean_images).cpu().numpy()
            latent_vectors.append(latent)
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


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
    title: str | None = None,
    y_label: str = "Loss",
) -> None:
    """Plots training vs validation loss curves across epochs."""
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


def plot_results(results_dict: dict[int, dict[str, float]], save_path: Path) -> None:
    """Plots final results (Mean Valid MSE vs Latent Dim)."""
    latent_dims = sorted(results_dict)
    mean_losses = [results_dict[dim]["mean_loss"] for dim in latent_dims]
    std_losses = [results_dict[dim]["std_loss"] for dim in latent_dims]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        latent_dims,
        mean_losses,
        yerr=std_losses,
        marker="o",
        linestyle="-",
        capsize=5,
    )
    axis.set_title("Compression vs Reconstruction Error")
    axis.set_xlabel("Latent Dimension")
    axis.set_ylabel("Mean Validation MSE")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_model_metric_comparison(
    rows: list[dict[str, float | int | str]],
    dataset_name: str,
    capacity: int,
    save_path: Path,
) -> None:
    """Compares different models on key metrics (MSE, PSNR, SSIM) for a single latent dimension."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    metric_specs = [
        ("Mean MSE", "mean_loss", "tab:red"),
        ("PSNR", "psnr", "tab:blue"),
        ("SSIM", "ssim", "tab:green"),
    ]

    for axis, (title, metric_key, color) in zip(axes, metric_specs):
        metric_rows = [
            row for row in rows
            if is_finite_metric(float(row[metric_key]))
        ]
        if not metric_rows:
            axis.set_title(title)
            axis.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
            axis.set_xticks([])
            axis.set_yticks([])
            continue

        model_labels = [str(row["model_type"]).upper() for row in metric_rows]
        values = [float(row[metric_key]) for row in metric_rows]
        x = np.arange(len(metric_rows))
        axis.bar(x, values, color=color, alpha=0.85)
        axis.set_title(title)
        axis.set_xticks(x, model_labels)
        axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle(f"{dataset_name.upper()} Model Comparison (Capacity = {capacity})")
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_metrics_vs_latent_by_model(
    rows: list[dict[str, float | int | str]],
    dataset_name: str,
    save_path: Path,
) -> None:
    """Plots metrics (MSE, PSNR, SSIM) vs Latent Dimension, grouped by model type."""
    metrics = [
        ("mean_loss", "Mean MSE"),
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
    ]
    model_styles = {
        "ae": ("AE", "tab:blue"),
        "dae": ("DAE", "tab:orange"),
        "vae": ("VAE", "tab:green"),
        "diffusion": ("Diffusion", "tab:purple"),
    }

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for axis, (metric_key, metric_label) in zip(axes, metrics):
        metric_has_values = False
        for model_key, (model_label, color) in model_styles.items():
            model_rows = [row for row in rows if str(row["model_type"]) == model_key]
            model_rows.sort(key=lambda row: int(row["latent_dim"]))
            if not model_rows:
                continue

            filtered_rows = [
                row for row in model_rows
                if is_finite_metric(float(row[metric_key]))
            ]
            if not filtered_rows:
                continue

            latent_dims = [int(row["latent_dim"]) for row in filtered_rows]
            metric_values = [float(row[metric_key]) for row in filtered_rows]
            axis.plot(
                latent_dims,
                metric_values,
                marker="o",
                linewidth=2,
                label=model_label,
                color=color,
            )
            metric_has_values = True

        axis.set_title(f"{metric_label} vs Model Capacity")
        axis.set_xlabel("Model Capacity")
        axis.set_ylabel(metric_label)
        axis.set_xticks(sorted({int(row["latent_dim"]) for row in rows}))
        axis.grid(True, alpha=0.3)
        if not metric_has_values:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axis.transAxes, fontsize=12)

    axes[0].legend()
    figure.suptitle(f"{dataset_name.upper()} Metrics by Model Capacity")
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_results_csv(results: list[dict[str, float | int | str]], save_path: Path) -> None:
    """Saves raw evaluation metrics to a CSV for external analysis or presentation tables."""
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "dataset_name",
                "model_type",
                "latent_dim",
                "mean_loss",
                "std_loss",
                "mean_psnr",
                "mean_ssim",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["dataset_name"],
                    row["model_type"],
                    row["latent_dim"],
                    row["mean_loss"],
                    row["std_loss"],
                    row["psnr"],
                    row["ssim"],
                ]
            )


# ==============================================================================
# 6. EXPERIMENT LOGIC
# ==============================================================================

def run_kfold_experiment(
    latent_dim: int,
    dataset_name: str = "mnist",
    model_type: str = "ae",
    noise_level: float = 0.2,
) -> dict[str, object]:
    """
    Runs a full K-Fold cross validation for a specific configuration.
    Returns metrics and best fold's parameters.
    """
    noise_text = f" | noise={noise_level:g}" if is_denoising(model_type) else ""
    log_prefix = f"[{dataset_name.upper()} | {model_type.upper()}{noise_text}]"
    capacity_label = format_capacity_label(model_type, latent_dim)
    split_label = "Split" if is_diffusion(model_type) else "Fold"

    fold_metrics: list[dict[str, float]] = []
    best_fold_payload: dict[str, object] | None = None

    if is_diffusion(model_type):
        train_dataset = get_dataset(dataset_name=dataset_name, train=True)
        test_dataset = get_dataset(dataset_name=dataset_name, train=False)
        train_loader = create_loader(train_dataset, shuffle=True)
        eval_loader = create_loader(test_dataset, shuffle=False)
        experiment_splits = [(1, train_loader, eval_loader)]

        print(f"{log_prefix} {capacity_label} | Train dataset size: {len(train_dataset)}")
        print(f"{log_prefix} {capacity_label} | Test dataset size: {len(test_dataset)}")
        print(f"{log_prefix} {capacity_label} | Train batches: {len(train_loader)}")
        print(f"{log_prefix} {capacity_label} | Test batches: {len(eval_loader)}")
    else:
        dataset = get_dataset(dataset_name=dataset_name, train=True)
        experiment_splits = []
        for fold_idx, train_indices, val_indices in build_experiment_splits(len(dataset), model_type):
            train_subset = Subset(dataset, train_indices.tolist())
            val_subset = Subset(dataset, val_indices.tolist())
            train_loader = create_loader(train_subset, shuffle=True)
            eval_loader = create_loader(val_subset, shuffle=False)
            experiment_splits.append((fold_idx, train_loader, eval_loader, train_indices, val_indices))

    for split_entry in experiment_splits:
        if is_diffusion(model_type):
            fold_idx, train_loader, eval_loader = split_entry
            val_indices = np.arange(len(eval_loader.dataset))
        else:
            fold_idx, train_loader, eval_loader, _, val_indices = split_entry

        # Extensible instantiator
        scheduler = None
        if is_vae_model(model_type):
            model_class = VariationalAutoencoder
            model = model_class(latent_dim=latent_dim).to(DEVICE)
        elif model_type in ("ae", "dae"):
            model_class = FullyConnectedAutoencoder
            model = model_class(latent_dim=latent_dim).to(DEVICE)
        elif model_type == "diffusion":
            # Diffusion reuses the experiment loop, but the model predicts added noise
            # rather than reconstructing the input image directly.
            model = DiffusionUNet(base_channels=latent_dim).to(DEVICE)
            scheduler = get_noise_schedule(CONFIG.diffusion_timesteps, DEVICE)
        else:
            raise NotImplementedError(f"Model initialization for {model_type} not implemented")
            
        optimizer = Adam(model.parameters(), lr=CONFIG.learning_rate)

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, CONFIG.epochs + 1):
            if is_diffusion(model_type):
                train_loss = train_diffusion_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    DEVICE,
                    progress_label=f"{log_prefix} {capacity_label} | {split_label} {fold_idx}",
                    progress_interval=CONFIG.diffusion_log_interval,
                )
                val_loss = eval_diffusion_epoch(
                    model,
                    eval_loader,
                    scheduler,
                    DEVICE,
                    progress_label=f"{log_prefix} {capacity_label} | {split_label} {fold_idx}",
                    progress_interval=CONFIG.diffusion_log_interval,
                    eval_split_name="Test",
                )
            else:
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    CRITERION,
                    DEVICE,
                    model_type=model_type,
                    noise_level=noise_level,
                )
                val_loss = eval_epoch(
                    model,
                    eval_loader,
                    CRITERION,
                    DEVICE,
                    model_type=model_type,
                    noise_level=noise_level,
                )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"{log_prefix} {capacity_label} | {split_label} {fold_idx} | "
                f"Epoch {epoch}/{CONFIG.epochs} | "
                f"Train Loss={train_loss:.6f} | {'Test' if is_diffusion(model_type) else 'Val'} Loss={val_loss:.6f}"
            )

        if model_type == "diffusion":
            metrics = evaluate_diffusion_metrics(model, eval_loader, scheduler, DEVICE)
            metrics["noise_mse"] = val_losses[-1]
        else:
            metrics = evaluate_metrics(model, eval_loader, CRITERION, DEVICE, model_type=model_type)
            
        fold_metrics.append(metrics)
        if model_type == "diffusion":
            print(
                f"{log_prefix} {capacity_label} | {split_label} {fold_idx} | "
                f"Noise MSE={metrics['noise_mse']:.6f}, Recon MSE={metrics['mse']:.6f}, "
                f"PSNR={format_metric(metrics['psnr'])}, SSIM={format_metric(metrics['ssim'])}"
            )
        else:
            print(
                f"{log_prefix} {capacity_label} | {split_label} {fold_idx} | "
                f"MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}"
            )

        # Track the best model across folds based on Mean Squared Error
        comparison_mse = metrics["noise_mse"] if model_type == "diffusion" else metrics["mse"]
        best_comparison_mse = (
            best_fold_payload["metrics"].get("noise_mse", best_fold_payload["metrics"]["mse"])
            if best_fold_payload is not None
            else None
        )
        if best_fold_payload is None or comparison_mse < best_comparison_mse:
            best_fold_payload = {
                "metrics": metrics,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "val_indices": val_indices.tolist(),
                "uses_test_split": is_diffusion(model_type),
                "train_losses": train_losses,
                "val_losses": val_losses,
            }

    all_losses = [entry.get("noise_mse", entry["mse"]) for entry in fold_metrics]
    all_psnr = [entry["psnr"] for entry in fold_metrics]
    all_ssim = [entry["ssim"] for entry in fold_metrics]
    
    mean_loss = float(np.mean(all_losses))
    std_loss = float(np.std(all_losses))
    mean_psnr = mean_metric(all_psnr)
    mean_ssim = mean_metric(all_ssim)

    if model_type == "diffusion":
        print(
            f"{log_prefix} {capacity_label} | Mean Noise Loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"Mean PSNR={format_metric(mean_psnr)} | Mean SSIM={format_metric(mean_ssim)}"
        )
    else:
        print(
            f"{log_prefix} {capacity_label} | Mean Loss={mean_loss:.6f} ± {std_loss:.6f} | "
            f"Mean PSNR={format_metric(mean_psnr)} | Mean SSIM={format_metric(mean_ssim)}"
        )

    return {
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "all_losses": all_losses,
        "fold_metrics": fold_metrics,
        "best_fold": best_fold_payload,
    }


def main() -> None:
    global CONFIG
    args = parse_args()
    selected_models = args.models
    selected_datasets = args.datasets
    config_updates: dict[str, object] = {}
    if args.epochs is not None:
        config_updates["epochs"] = args.epochs
    if args.batch_size is not None:
        config_updates["batch_size"] = args.batch_size
    if args.n_splits is not None:
        config_updates["n_splits"] = args.n_splits
    if args.base_channels is not None:
        config_updates["latent_dims"] = tuple(args.base_channels)
    if args.diffusion_timesteps is not None:
        config_updates["diffusion_timesteps"] = args.diffusion_timesteps
    if args.diffusion_log_interval is not None:
        config_updates["diffusion_log_interval"] = args.diffusion_log_interval
    if args.diffusion_val_fraction is not None:
        config_updates["diffusion_val_fraction"] = args.diffusion_val_fraction
    if config_updates:
        CONFIG = replace(CONFIG, **config_updates)

    set_seed(CONFIG.random_seed)
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    diffusion_output_dirs = ensure_diffusion_output_dirs(CONFIG.output_dir)

    print(f"Using device: {device}")
    print(f"Selected datasets: {', '.join(dataset.upper() for dataset in selected_datasets)}")
    print(f"Selected models: {', '.join(model.upper() for model in selected_models)}")
    all_results: list[dict[str, float | int | str]] = []
    
    for dataset_name in selected_datasets:
        dataset = get_dataset(dataset_name=dataset_name, train=True)
        print(f"Dataset {dataset_name.upper()} train size: {len(dataset)} samples")

        for latent_dim in CONFIG.latent_dims:
            for model_type in selected_models:
                capacity_label = format_capacity_label(model_type, latent_dim)
                
                noise_level = DAE_NOISE_LEVEL if is_denoising(model_type) else 0.0
                noise_label = f"{noise_level:g}"
                
                if model_type == "diffusion":
                    print()
                    print_diffusion_experiment_header(dataset_name, latent_dim)
                elif is_denoising(model_type):
                    print(f"\nRunning {model_type.upper()} with noise={noise_label}")
                    print(f"[{dataset_name.upper()} | {model_type.upper()} | noise={noise_label}] {capacity_label}")
                else:
                    print(f"\nRunning {model_type.upper()}")
                    print(f"[{dataset_name.upper()} | {model_type.upper()}] {capacity_label}")

                experiment_result = run_kfold_experiment(
                    latent_dim,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    noise_level=noise_level,
                )
                
                row: dict[str, float | int | str] = {
                    "dataset_name": dataset_name,
                    "model_type": model_type,
                    "latent_dim": latent_dim,
                    "mean_loss": float(experiment_result["mean_loss"]),
                    "std_loss": float(experiment_result["std_loss"]),
                    "psnr": float(experiment_result["psnr"]),
                    "ssim": float(experiment_result["ssim"]),
                }
                if is_denoising(model_type):
                    row["noise_level"] = noise_label
                all_results.append(row)

                best_fold = experiment_result["best_fold"]
                if best_fold is None:
                    continue

                scheduler = None
                if is_vae_model(model_type):
                    model_class = VariationalAutoencoder
                    model = model_class(latent_dim=latent_dim).to(DEVICE)
                elif model_type in ("ae", "dae"):
                    model_class = FullyConnectedAutoencoder
                    model = model_class(latent_dim=latent_dim).to(DEVICE)
                elif model_type == "diffusion":
                    model = DiffusionUNet(base_channels=latent_dim).to(DEVICE)
                    scheduler = get_noise_schedule(CONFIG.diffusion_timesteps, DEVICE)
                else:
                    raise NotImplementedError(f"Model visualization unsupported for {model_type}")
                    
                model.load_state_dict(best_fold["model_state_dict"])
                
                if model_type == "diffusion":
                    best_eval_dataset = get_dataset(dataset_name=dataset_name, train=False)
                    best_val_loader = create_loader(best_eval_dataset, shuffle=False)
                else:
                    best_val_subset = Subset(dataset, best_fold["val_indices"])
                    best_val_loader = create_loader(best_val_subset, shuffle=False)
                
                artifact_stem = f"{dataset_name}_{model_type}"
                if is_denoising(model_type):
                    artifact_stem = f"{artifact_stem}_noise_{noise_label}"

                if model_type == "diffusion":
                    diffusion_config = build_diffusion_artifact_config(dataset_name, latent_dim)
                    plot_loss_curves(
                        best_fold["train_losses"],
                        best_fold["val_losses"],
                        diffusion_output_dirs["loss_curves"]
                        / f"{format_experiment_name(diffusion_config, 'diffusion_loss', include_epochs=True)}.png",
                        title=(
                            f"Diffusion Training Loss "
                            f"(Base Channels = {latent_dim}, Timesteps = {CONFIG.diffusion_timesteps})"
                        ),
                        y_label="MSE Loss",
                    )
                    generated_samples = sample_images(
                        model,
                        scheduler,
                        DEVICE,
                        num_samples=10,
                    )
                    plot_image_grid(
                        generated_samples,
                        diffusion_output_dirs["samples"]
                        / f"{format_experiment_name(diffusion_config, 'samples')}.png",
                        title=f"Diffusion Generated Samples ({dataset_name.upper()}, Base Channels = {latent_dim})",
                    )
                    plot_diffusion_snapshots(
                        model,
                        scheduler,
                        DEVICE,
                        dataset_name=dataset_name,
                        base_channels=latent_dim,
                        save_path=diffusion_output_dirs["snapshots"]
                        / f"{format_experiment_name(diffusion_config, 'snapshots')}.png",
                    )
                else:
                    plot_loss_curves(
                        best_fold["train_losses"],
                        best_fold["val_losses"],
                        CONFIG.output_dir / f"{artifact_stem}_loss_curve_{latent_dim}.png",
                    )
                    show_reconstructions(
                        model,
                        best_val_loader,
                        DEVICE,
                        CONFIG.output_dir / f"{artifact_stem}_latent_{latent_dim}.png",
                        model_type=model_type,
                        noise_level=noise_level,
                    )
                    plot_latent_space(
                        model,
                        best_val_loader,
                        DEVICE,
                        CONFIG.output_dir / f"{artifact_stem}_latent_space_{latent_dim}.png",
                    )
                    
                    if is_vae_model(model_type):
                        generate_samples(
                            model,
                            latent_dim,
                            DEVICE,
                            save_path=CONFIG.output_dir / f"vae_generated_{dataset_name}_latent_{latent_dim}.png",
                        )
                        interpolate_images(
                            model,
                            best_val_loader,
                            DEVICE,
                            save_path=CONFIG.output_dir / f"vae_interpolation_{dataset_name}_latent_{latent_dim}.png",
                        )

            latent_rows = [
                row
                for row in all_results
                if row["dataset_name"] == dataset_name and int(row["latent_dim"]) == latent_dim
            ]
            plot_model_metric_comparison(
                latent_rows,
                dataset_name,
                latent_dim,
                CONFIG.output_dir / f"{dataset_name}_model_comparison_capacity_{latent_dim}.png",
            )

        dataset_rows = [row for row in all_results if row["dataset_name"] == dataset_name]
        plot_metrics_vs_latent_by_model(
            dataset_rows,
            dataset_name,
            CONFIG.output_dir / f"{dataset_name}_metrics_vs_capacity_by_model.png",
        )

    save_results_csv(all_results, CONFIG.output_dir / "kfold_results.csv")

    print("\nFinal Summary")
    for row in all_results:
        noise_suffix = f" | noise={row['noise_level']}" if "noise_level" in row else ""
        capacity_label = format_capacity_label(str(row["model_type"]), int(row["latent_dim"]))
        loss_label = "Mean Noise Loss" if str(row["model_type"]) == "diffusion" else "Mean Loss"
        print(
            f"[{str(row['dataset_name']).upper()} | {str(row['model_type']).upper()}{noise_suffix}] "
            f"{capacity_label} | {loss_label}={float(row['mean_loss']):.6f} ± "
            f"{float(row['std_loss']):.6f} | PSNR={format_metric(float(row['psnr']))} | "
            f"SSIM={format_metric(float(row['ssim']))}"
        )

    print(f"\nArtifacts saved to: {CONFIG.output_dir.resolve()}")


if __name__ == "__main__":
    main()
