"""Autoencoder models, training loops, metrics, and report artifacts."""

from autoencoders.artifacts import (
    generate_samples,
    interpolate_images,
    plot_latent_space,
    show_reconstructions,
)
from autoencoders.data import (
    SUPPORTED_AUTOENCODER_DATASET_CHOICES,
    build_autoencoder_dataset,
    normalize_autoencoder_dataset_name,
    resolve_autoencoder_dataset_spec,
)
from autoencoders.models import FullyConnectedAutoencoder, VariationalAutoencoder
from autoencoders.training import (
    eval_epoch,
    evaluate_metrics,
    is_denoising,
    is_vae_model,
    train_epoch,
)

__all__ = [
    "FullyConnectedAutoencoder",
    "SUPPORTED_AUTOENCODER_DATASET_CHOICES",
    "VariationalAutoencoder",
    "build_autoencoder_dataset",
    "eval_epoch",
    "evaluate_metrics",
    "generate_samples",
    "interpolate_images",
    "is_denoising",
    "is_vae_model",
    "normalize_autoencoder_dataset_name",
    "plot_latent_space",
    "resolve_autoencoder_dataset_spec",
    "show_reconstructions",
    "train_epoch",
]
