from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class AutoencoderDatasetSpec:
    """Dataset metadata for AE/DAE/VAE runs."""

    name: str
    aliases: tuple[str, ...]
    dataset_class: type[datasets.VisionDataset]
    num_classes: int = 10
    native_image_size: int = 28
    native_channels: int = 1


AUTOENCODER_DATASET_SPECS: dict[str, AutoencoderDatasetSpec] = {
    "mnist": AutoencoderDatasetSpec(
        name="mnist",
        aliases=("mnist",),
        dataset_class=datasets.MNIST,
    ),
    "fashion": AutoencoderDatasetSpec(
        name="fashion",
        aliases=("fashion", "fashion-mnist", "fashion_mnist"),
        dataset_class=datasets.FashionMNIST,
    ),
}

AUTOENCODER_DATASET_ALIASES: dict[str, str] = {
    alias: spec.name
    for spec in AUTOENCODER_DATASET_SPECS.values()
    for alias in spec.aliases
}
SUPPORTED_AUTOENCODER_DATASET_CHOICES: tuple[str, ...] = tuple(AUTOENCODER_DATASET_ALIASES)


def normalize_autoencoder_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.lower()
    if normalized not in AUTOENCODER_DATASET_ALIASES:
        raise ValueError(f"Unsupported autoencoder dataset: {dataset_name}")
    return AUTOENCODER_DATASET_ALIASES[normalized]


def resolve_autoencoder_dataset_spec(dataset_name: str) -> AutoencoderDatasetSpec:
    return AUTOENCODER_DATASET_SPECS[normalize_autoencoder_dataset_name(dataset_name)]


def build_autoencoder_dataset(
    dataset_name: str,
    *,
    root: Path,
    train: bool,
    download: bool = False,
) -> Dataset:
    """Instantiate an AE/DAE/VAE dataset with the original 28x28 grayscale transform."""

    spec = resolve_autoencoder_dataset_spec(dataset_name)
    dataset_root = Path(root)
    if download:
        dataset_root.mkdir(parents=True, exist_ok=True)
    return spec.dataset_class(
        root=str(dataset_root),
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )
