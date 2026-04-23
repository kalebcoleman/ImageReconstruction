from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class DatasetSpec:
    """Static dataset metadata used to build compatible training adapters."""

    name: str
    aliases: tuple[str, ...]
    dataset_class: type[datasets.VisionDataset] | None
    num_classes: int
    native_image_size: int
    native_channels: int
    train_split: str = "train"
    eval_split: str = "test"
    supports_download: bool = True
    storage_subdir: str | None = None

    def split_name(self, train: bool) -> str:
        return self.train_split if train else self.eval_split


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        aliases=("mnist",),
        dataset_class=datasets.MNIST,
        num_classes=10,
        native_image_size=28,
        native_channels=1,
    ),
    "fashion": DatasetSpec(
        name="fashion",
        aliases=("fashion", "fashion-mnist", "fashion_mnist"),
        dataset_class=datasets.FashionMNIST,
        num_classes=10,
        native_image_size=28,
        native_channels=1,
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        aliases=("cifar10", "cifar", "cifar-10", "cifar_10"),
        dataset_class=datasets.CIFAR10,
        num_classes=10,
        native_image_size=32,
        native_channels=3,
    ),
    "imagenet": DatasetSpec(
        name="imagenet",
        aliases=("imagenet", "ilsvrc", "ilsvrc2012"),
        dataset_class=None,
        num_classes=1000,
        native_image_size=224,
        native_channels=3,
        eval_split="val",
        supports_download=False,
        storage_subdir="imagenet",
    ),
}

DATASET_ALIASES: dict[str, str] = {
    alias: spec.name
    for spec in DATASET_SPECS.values()
    for alias in spec.aliases
}
SUPPORTED_DATASET_CHOICES: tuple[str, ...] = tuple(DATASET_ALIASES)
SUPPORTED_PREPROCESSING_PROTOCOLS: tuple[str, ...] = ("default", "parity_64")


@dataclass(frozen=True)
class ResolvedDiffusionDataConfig:
    """Resolved image-shape metadata for the active diffusion run."""

    image_size: int
    channels: int
    num_classes: int


def normalize_dataset_name(dataset_name: str) -> str:
    """Resolve any supported alias into the repo's canonical dataset key."""

    normalized = dataset_name.lower()
    if normalized not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_ALIASES[normalized]


def resolve_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Return the canonical dataset metadata for a user-provided name."""

    return DATASET_SPECS[normalize_dataset_name(dataset_name)]


def resolve_diffusion_data_config(
    dataset_name: str,
    *,
    diffusion_backbone: str,
    image_size: int | None,
    channels: int | None,
) -> ResolvedDiffusionDataConfig:
    """Resolve the diffusion image shape for a dataset/backbone combination."""

    spec = resolve_dataset_spec(dataset_name)
    resolved_image_size = image_size
    if resolved_image_size is None:
        resolved_image_size = 64 if diffusion_backbone == "adm" else spec.native_image_size

    resolved_channels = channels
    if resolved_channels is None:
        resolved_channels = 3 if diffusion_backbone == "adm" else spec.native_channels

    if resolved_image_size < 8:
        raise ValueError("Diffusion image_size must be at least 8.")
    if resolved_channels < 1:
        raise ValueError("Diffusion channels must be at least 1.")

    return ResolvedDiffusionDataConfig(
        image_size=resolved_image_size,
        channels=resolved_channels,
        num_classes=spec.num_classes,
    )


def build_diffusion_transform(
    dataset_name: str,
    *,
    train: bool,
    image_size: int,
    channels: int,
    preprocessing_protocol: str = "default",
) -> transforms.Compose:
    """Build the diffusion preprocessing pipeline for a dataset."""

    spec = resolve_dataset_spec(dataset_name)
    steps: list[transforms.Transform] = []
    if preprocessing_protocol not in SUPPORTED_PREPROCESSING_PROTOCOLS:
        raise ValueError(
            f"Unsupported preprocessing protocol: {preprocessing_protocol}. "
            f"Expected one of {SUPPORTED_PREPROCESSING_PROTOCOLS}."
        )

    if preprocessing_protocol == "parity_64":
        if spec.name == "imagenet":
            resize_size = max(image_size, math.ceil(image_size * 1.125))
            steps.extend(
                [
                    transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(image_size),
                ]
            )
        else:
            steps.append(
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BILINEAR,
                )
            )
    else:
        if spec.name == "imagenet":
            if train:
                steps.extend(
                    [
                        transforms.RandomResizedCrop(
                            image_size,
                            interpolation=InterpolationMode.BILINEAR,
                        ),
                        transforms.RandomHorizontalFlip(),
                    ]
                )
            else:
                resize_size = max(image_size, math.ceil(image_size * 1.125))
                steps.extend(
                    [
                        transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
                        transforms.CenterCrop(image_size),
                    ]
                )
        else:
            steps.append(
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BILINEAR,
                )
            )
            if train and spec.name in {"cifar10"}:
                steps.append(transforms.RandomHorizontalFlip())

    if spec.native_channels != channels:
        steps.append(transforms.Grayscale(num_output_channels=channels))

    steps.append(transforms.ToTensor())
    mean = tuple(0.5 for _ in range(channels))
    std = tuple(0.5 for _ in range(channels))
    steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def describe_diffusion_preprocessing(
    dataset_name: str,
    *,
    image_size: int,
    channels: int,
    preprocessing_protocol: str,
) -> dict[str, object]:
    """Return an explicit, serializable description of the diffusion transforms."""

    spec = resolve_dataset_spec(dataset_name)
    base_description: dict[str, object] = {
        "dataset": spec.name,
        "protocol": preprocessing_protocol,
        "image_size": image_size,
        "channels": channels,
        "channel_conversion": (
            f"{spec.native_channels}->" f"{channels}"
            if spec.native_channels != channels
            else f"{channels}->" f"{channels}"
        ),
        "normalization": {
            "range_in": "[0, 1]",
            "range_out": "[-1, 1]",
            "mean": [0.5 for _ in range(channels)],
            "std": [0.5 for _ in range(channels)],
        },
    }

    if preprocessing_protocol == "parity_64":
        if spec.name == "imagenet":
            train_ops = [
                f"resize({max(image_size, math.ceil(image_size * 1.125))})",
                f"center_crop({image_size})",
            ]
        else:
            train_ops = [f"resize({image_size}x{image_size})"]
        eval_ops = list(train_ops)
        base_description["deterministic_train_preprocessing"] = True
    else:
        if spec.name == "imagenet":
            train_ops = [f"random_resized_crop({image_size})", "random_horizontal_flip"]
            eval_ops = [f"resize({max(image_size, math.ceil(image_size * 1.125))})", f"center_crop({image_size})"]
        else:
            train_ops = [f"resize({image_size}x{image_size})"]
            if spec.name == "cifar10":
                train_ops.append("random_horizontal_flip")
            eval_ops = [f"resize({image_size}x{image_size})"]
        base_description["deterministic_train_preprocessing"] = False

    base_description["train_ops"] = train_ops
    base_description["eval_ops"] = eval_ops
    return base_description


def build_standard_transform(dataset_name: str) -> transforms.Compose:
    """Return the unchanged AE/DAE/VAE preprocessing pipeline."""

    spec = resolve_dataset_spec(dataset_name)
    if spec.native_channels != 1 or spec.native_image_size != 28:
        raise ValueError(
            f"{spec.name} is only supported for diffusion in the current repo. "
            "AE/DAE/VAE remain MNIST-style paths for now."
        )
    return transforms.Compose([transforms.ToTensor()])


def build_dataset(
    dataset_name: str,
    *,
    root: Path,
    train: bool,
    diffusion: bool,
    image_size: int | None = None,
    channels: int | None = None,
    preprocessing_protocol: str = "default",
    download: bool = False,
) -> Dataset:
    """Instantiate a dataset with the correct split and transform adapter."""

    spec = resolve_dataset_spec(dataset_name)
    dataset_root = Path(root)

    if diffusion:
        if image_size is None or channels is None:
            raise ValueError("Diffusion datasets require an explicit image_size and channels.")
        transform = build_diffusion_transform(
            spec.name,
            train=train,
            image_size=image_size,
            channels=channels,
            preprocessing_protocol=preprocessing_protocol,
        )
    else:
        transform = build_standard_transform(spec.name)

    if spec.name == "imagenet":
        if download:
            raise ValueError(
                "ImageNet download is not supported. Prepare "
                f"{(dataset_root / (spec.storage_subdir or spec.name)).resolve()} "
                "with train/ and val/ subdirectories."
            )
        split_dir = dataset_root / (spec.storage_subdir or spec.name) / spec.split_name(train)
        if not split_dir.exists():
            raise FileNotFoundError(
                "ImageNet data was not found. Expected a directory like "
                f"{split_dir.resolve()} containing class subdirectories."
            )
        return datasets.ImageFolder(root=str(split_dir), transform=transform)

    if spec.dataset_class is None:  # pragma: no cover - guarded by the ImageNet branch above.
        raise ValueError(f"No dataset class registered for {spec.name}.")

    if download and spec.supports_download:
        dataset_root.mkdir(parents=True, exist_ok=True)

    return spec.dataset_class(
        root=str(dataset_root),
        train=train,
        download=download,
        transform=transform,
    )
