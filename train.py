from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "image_reconstruction_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
from diffusion.sampling import sample_images
from diffusion.scheduler import get_noise_schedule
from diffusion.training import (
    _compute_batch_ssim,
    eval_diffusion_epoch,
    evaluate_diffusion_metrics,
    train_diffusion_epoch,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("image_reconstruction")
CRITERION = nn.MSELoss()
DATASET_TRANSFORM = transforms.ToTensor()
DATASET_CACHE: dict[tuple[str, bool, str, bool], datasets.VisionDataset] = {}
SUPPORTED_MODELS = ("ae", "dae", "vae", "diffusion")
SUPPORTED_DATASETS: dict[str, type[datasets.VisionDataset]] = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "fashion-mnist": datasets.FashionMNIST,
    "fashion_mnist": datasets.FashionMNIST,
}


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "all"
    dataset: str = "mnist"
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 42
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    run_name: str | None = None
    num_workers: int = 0
    download: bool = False
    latent_dim: int = 8
    timesteps: int = 10
    base_channels: int = 8
    time_dim: int = 64
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    sample_count: int = 10
    n_splits: int = 2
    dae_noise_level: float = 0.2
    diffusion_log_interval: int = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image reconstruction models.")
    parser.add_argument(
        "--model",
        choices=(*SUPPORTED_MODELS, "all"),
        default="all",
        help="Model to train. Use 'all' to run the historical multi-model sweep.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        help="Compatibility alias for the old multi-model sweep interface.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(SUPPORTED_DATASETS),
        default="mnist",
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SUPPORTED_DATASETS),
        help="Compatibility alias for the old multi-dataset sweep interface.",
    )
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=ExperimentConfig.lr, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", type=Path, default=ExperimentConfig.data_dir)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument("--run_name", "--run-name", dest="run_name", default=None)
    parser.add_argument("--num_workers", "--num-workers", "--workers", dest="num_workers", type=int, default=ExperimentConfig.num_workers)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=ExperimentConfig.download,
        help="Allow dataset downloads when files are missing. Defaults to false for cluster safety.",
    )
    parser.add_argument("--latent_dim", "--latent-dim", dest="latent_dim", type=int, default=ExperimentConfig.latent_dim)
    parser.add_argument(
        "--timesteps",
        "--diffusion-timesteps",
        "--diffusion_timesteps",
        dest="timesteps",
        type=int,
        default=ExperimentConfig.timesteps,
    )
    parser.add_argument(
        "--base_channels",
        "--base-channels",
        "--diffusion-base-channels",
        "--model-width",
        dest="base_channels",
        type=int,
        default=ExperimentConfig.base_channels,
    )
    parser.add_argument("--time_dim", "--time-dim", dest="time_dim", type=int, default=ExperimentConfig.time_dim)
    parser.add_argument("--beta_start", "--beta-start", dest="beta_start", type=float, default=ExperimentConfig.beta_start)
    parser.add_argument("--beta_end", "--beta-end", dest="beta_end", type=float, default=ExperimentConfig.beta_end)
    parser.add_argument("--sample_count", "--sample-count", dest="sample_count", type=int, default=ExperimentConfig.sample_count)
    parser.add_argument("--n_splits", "--n-splits", dest="n_splits", type=int, default=ExperimentConfig.n_splits)
    parser.add_argument(
        "--diffusion_log_interval",
        "--diffusion-log-interval",
        dest="diffusion_log_interval",
        type=int,
        default=ExperimentConfig.diffusion_log_interval,
    )
    parser.add_argument(
        "--dae_noise_level",
        "--dae-noise-level",
        dest="dae_noise_level",
        type=float,
        default=ExperimentConfig.dae_noise_level,
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.n_splits < 2:
        parser.error("--n_splits must be at least 2")
    if args.timesteps < 1:
        parser.error("--timesteps must be at least 1")
    if args.base_channels < 1 or args.latent_dim < 1 or args.time_dim < 1:
        parser.error("--base_channels, --latent_dim, and --time_dim must be at least 1")
    if args.beta_start <= 0 or args.beta_end <= 0 or args.beta_end <= args.beta_start:
        parser.error("--beta_end must be greater than --beta_start, and both must be positive")
    if args.sample_count < 1:
        parser.error("--sample_count must be at least 1")
    if args.dae_noise_level < 0:
        parser.error("--dae_noise_level cannot be negative")
    if args.diffusion_log_interval < 0:
        parser.error("--diffusion_log_interval cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    return args


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.lower()
    if normalized in {"fashion_mnist", "fashion-mnist"}:
        return "fashion"
    return normalized


def resolve_selected_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return list(args.models)
    if args.model == "all":
        return list(SUPPORTED_MODELS)
    return [args.model]


def resolve_selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return [normalize_dataset_name(name) for name in args.datasets]
    return [normalize_dataset_name(args.dataset)]


def build_base_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        model=args.model,
        dataset=normalize_dataset_name(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_workers=args.num_workers,
        download=args.download,
        latent_dim=args.latent_dim,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        sample_count=args.sample_count,
        n_splits=args.n_splits,
        dae_noise_level=args.dae_noise_level,
        diffusion_log_interval=args.diffusion_log_interval,
    )


def build_run_config(base_config: ExperimentConfig, dataset_name: str, model_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        **{
            **asdict(base_config),
            "dataset": dataset_name,
            "model": model_name,
            "data_dir": Path(base_config.data_dir),
            "output_dir": Path(base_config.output_dir),
        }
    )


def slugify(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "run"


def format_run_float(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-2 or magnitude >= 1e3:
        text = f"{value:.0e}"
        text = re.sub(r"e([+-])0*(\d+)", r"e\1\2", text)
        return text.replace("e+", "e")
    return f"{value:g}"


def auto_run_name(config: ExperimentConfig, timestamp: datetime) -> str:
    lr_tag = format_run_float(config.lr)
    time_tag = timestamp.strftime("%Y-%m-%d_%H%M%S_%f")
    if config.model == "diffusion":
        return (
            f"{config.dataset}_{config.model}_t{config.timesteps}_ch{config.base_channels}"
            f"_bs{config.batch_size}_lr{lr_tag}_seed{config.seed}_{time_tag}"
        )
    return (
        f"{config.dataset}_{config.model}_z{config.latent_dim}"
        f"_bs{config.batch_size}_lr{lr_tag}_seed{config.seed}_{time_tag}"
    )


def resolve_run_dir(config: ExperimentConfig, *, timestamp: datetime) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    base_dir = output_dir / config.dataset / config.model
    requested_name = slugify(config.run_name) if config.run_name else slugify(auto_run_name(config, timestamp))
    base_dir.mkdir(parents=True, exist_ok=True)

    suffix = 0
    while True:
        suffix_label = "" if suffix == 0 else f"_{suffix:02d}"
        candidate = base_dir / f"{requested_name}{suffix_label}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        break

    paths = {
        "root": candidate,
        "checkpoints": candidate / "checkpoints",
        "plots": candidate / "plots",
        "samples": candidate / "samples",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(run_dir: Path) -> Path:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)

    log_path = run_dir / "train.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False
    logging.captureWarnings(True)
    return log_path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def collect_runtime_environment() -> dict[str, str]:
    tracked_names = {
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TMPDIR",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
    }
    payload: dict[str, str] = {}
    for name in sorted(os.environ):
        if name.startswith("SLURM_") or name in tracked_names:
            payload[name] = os.environ[name]
    return payload


def detect_git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_config(config: ExperimentConfig, run_dir: Path, *, cli_args: list[str], resolved_paths: dict[str, Path]) -> Path:
    config_payload = json_ready(asdict(config))
    config_payload["cli_args"] = cli_args
    config_payload["resolved_paths"] = json_ready(resolved_paths)
    config_payload["hostname"] = socket.gethostname()
    config_payload["saved_at"] = datetime.now().isoformat()
    config_payload["environment"] = collect_runtime_environment()
    config_payload["python_version"] = sys.version
    config_payload["torch_version"] = torch.__version__
    config_payload["numpy_version"] = np.__version__
    git_commit = detect_git_commit(Path(__file__).resolve().parent)
    if git_commit is not None:
        config_payload["git_commit"] = git_commit
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    return config_path


def append_metrics_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), sort_keys=True))
        handle.write("\n")


def save_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def dataset_missing_error(dataset_name: str, data_dir: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"{dataset_name.upper()} was not found under {data_dir.resolve()}. "
        "Pre-download the dataset on a login node or rerun with --download."
    )


def get_dataset(config: ExperimentConfig, dataset_name: str, *, train: bool) -> datasets.VisionDataset:
    dataset_key = normalize_dataset_name(dataset_name)
    dataset_class = SUPPORTED_DATASETS.get(dataset_key)
    if dataset_class is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data_dir = Path(config.data_dir)
    cache_key = (dataset_key, train, str(data_dir.resolve()), config.download)
    if cache_key in DATASET_CACHE:
        return DATASET_CACHE[cache_key]

    if config.download:
        data_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = dataset_class(
            root=str(data_dir),
            train=train,
            download=config.download,
            transform=DATASET_TRANSFORM,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if not config.download and ("dataset not found" in message or "not found" in message):
            raise dataset_missing_error(dataset_key, data_dir) from exc
        raise

    DATASET_CACHE[cache_key] = dataset
    return dataset


def create_loader(dataset: Dataset, config: ExperimentConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def build_experiment_splits(dataset_size: int, config: ExperimentConfig) -> list[tuple[int, np.ndarray, np.ndarray]]:
    splitter = KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    return [
        (fold_idx, train_indices, val_indices)
        for fold_idx, (train_indices, val_indices) in enumerate(splitter.split(range(dataset_size)), start=1)
    ]


def compute_psnr(mse: float) -> float:
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def is_denoising(model_type: str) -> bool:
    return model_type == "dae"


def is_vae_model(model_type: str) -> bool:
    return model_type == "vae"


def is_diffusion(model_type: str) -> bool:
    return model_type == "diffusion"


def is_finite_metric(value: float) -> bool:
    return math.isfinite(value)


def format_metric(value: float, precision: int = 4) -> str:
    if not is_finite_metric(value):
        return "N/A"
    return f"{value:.{precision}f}"


def mean_metric(values: list[float]) -> float:
    finite_values = [value for value in values if is_finite_metric(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


class FullyConnectedAutoencoder(nn.Module):
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
        return decoded.view(-1, 1, 28, 28)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        return self.decode(latent)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        hidden_dim = 400
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self.flatten(x))
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_features(x)
        return mu

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        return decoded.view(-1, 1, 28, 28)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_features(x)
        return self.decode(mu)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_features(inputs)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent)
        return reconstructed, mu, logvar


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
                ssim_total += _compute_batch_ssim(reconstructed_images, clean_images)
            num_batches += 1

    mse = mse_total / max(num_batches, 1)
    return {
        "mse": mse,
        "psnr": compute_psnr(mse),
        "ssim": ssim_total / max(num_batches, 1),
    }


def prepare_display_images(images: torch.Tensor, *, rescale: bool = False) -> torch.Tensor:
    display_images = images.detach().cpu().float()
    if rescale:
        flat = display_images.reshape(display_images.shape[0], -1)
        mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
        display_images = (display_images - mins) / (maxs - mins).clamp_min(1e-8)
    return display_images.clamp(0.0, 1.0)


def plot_image_row(images: torch.Tensor, save_path: Path, title: str) -> None:
    num_images = images.shape[0]
    figure, axes = plt.subplots(1, num_images, figsize=(1.5 * num_images, 2.0))
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, images):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_image_grid(images: torch.Tensor, save_path: Path, title: str, *, num_cols: int = 5) -> None:
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


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
    *,
    title: str | None = None,
    y_label: str = "Loss",
) -> None:
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


def plot_latent_space(model: nn.Module, loader: DataLoader, device: torch.device, save_path: Path) -> None:
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
) -> None:
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

    if noisy_images is not None:
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


def generate_samples(
    model: nn.Module,
    latent_dim: int,
    device: torch.device,
    *,
    save_path: Path,
    num_samples: int,
) -> None:
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        generated = model.decoder(z).view(-1, 1, 28, 28).cpu()
    plot_image_row(generated, save_path, title="VAE Generated Samples")


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

    plot_image_row(interpolated, save_path, title="VAE Latent Interpolation")


def plot_diffusion_snapshots(
    model: nn.Module,
    scheduler: object,
    device: torch.device,
    *,
    dataset_name: str,
    base_channels: int,
    save_path: Path,
    num_samples: int,
    num_snapshots: int = 9,
) -> None:
    _, intermediate_images, intermediate_steps = sample_images(
        model,
        scheduler,
        device,
        num_samples=num_samples,
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
        display_images = prepare_display_images(images, rescale=True)
        for col_idx in range(num_samples):
            axis = axes[row_idx, col_idx]
            axis.imshow(display_images[col_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if col_idx == 0:
                axis.set_ylabel(f"t={step_num}", rotation=0, labelpad=24, va="center", fontsize=9, fontweight="bold")

    figure.suptitle(
        f"Reverse Diffusion Process (Noise -> Image)\n"
        f"{dataset_name.upper()} | Base Channels = {base_channels}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def instantiate_model(config: ExperimentConfig, device: torch.device) -> tuple[nn.Module, Any | None]:
    scheduler = None
    if is_vae_model(config.model):
        model = VariationalAutoencoder(latent_dim=config.latent_dim).to(device)
    elif config.model in ("ae", "dae"):
        model = FullyConnectedAutoencoder(latent_dim=config.latent_dim).to(device)
    elif is_diffusion(config.model):
        model = DiffusionUNet(base_channels=config.base_channels, time_dim=config.time_dim).to(device)
        scheduler = get_noise_schedule(
            config.timesteps,
            device,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
    else:
        raise NotImplementedError(f"Unsupported model: {config.model}")
    return model, scheduler


def log_run_header(config: ExperimentConfig, run_paths: dict[str, Path], device: torch.device, cli_args: list[str]) -> None:
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    data_dir = Path(config.data_dir)
    LOGGER.info("Start time: %s", datetime.now().isoformat())
    LOGGER.info("Hostname: %s", socket.gethostname())
    LOGGER.info("Device: %s (%s)", device, cuda_name)
    LOGGER.info("Seed: %s", config.seed)
    LOGGER.info("CLI args: %s", " ".join(cli_args) if cli_args else "<none>")
    LOGGER.info("Resolved output directory: %s", run_paths["root"].resolve())
    LOGGER.info("Data directory: %s", data_dir.resolve())
    LOGGER.info("Download enabled: %s", config.download)
    LOGGER.info(
        "Run config: dataset=%s model=%s epochs=%d batch_size=%d lr=%g num_workers=%d",
        config.dataset,
        config.model,
        config.epochs,
        config.batch_size,
        config.lr,
        config.num_workers,
    )
    if is_diffusion(config.model):
        LOGGER.info(
            "Model hyperparameters: base_channels=%d time_dim=%d timesteps=%d beta_start=%g beta_end=%g sample_count=%d",
            config.base_channels,
            config.time_dim,
            config.timesteps,
            config.beta_start,
            config.beta_end,
            config.sample_count,
        )
    else:
        LOGGER.info(
            "Model hyperparameters: latent_dim=%d dae_noise_level=%g n_splits=%d",
            config.latent_dim,
            config.dae_noise_level,
            config.n_splits,
        )


def save_checkpoint(path: Path, model: nn.Module, config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": json_ready(asdict(config)),
            "metrics": json_ready(metrics),
        },
        path,
    )


def save_run_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["dataset", "model", "mean_loss", "std_loss", "psnr", "ssim", "parameters", "seed"])
        writer.writerow(
            [
                summary["dataset"],
                summary["model"],
                summary["mean_loss"],
                summary["std_loss"],
                summary["psnr"],
                summary["ssim"],
                summary["model_parameters"],
                summary["seed"],
            ]
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_single_experiment(config: ExperimentConfig, cli_args: list[str], device: torch.device) -> Path:
    timestamp = datetime.now()
    run_paths = resolve_run_dir(config, timestamp=timestamp)
    setup_logging(run_paths["root"])
    seed_everything(config.seed)
    config_path = save_config(config, run_paths["root"], cli_args=cli_args, resolved_paths=run_paths)
    metrics_jsonl_path = run_paths["root"] / "metrics.jsonl"

    log_run_header(config, run_paths, device, cli_args)
    LOGGER.info("Saved resolved config to %s", config_path.resolve())

    noise_level = config.dae_noise_level if is_denoising(config.model) else 0.0

    fold_metrics: list[dict[str, float]] = []
    best_payload: dict[str, Any] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None

    if is_diffusion(config.model):
        train_dataset = get_dataset(config, config.dataset, train=True)
        eval_dataset = get_dataset(config, config.dataset, train=False)
        train_loader = create_loader(train_dataset, config, shuffle=True)
        eval_loader = create_loader(eval_dataset, config, shuffle=False)
        LOGGER.info("Dataset sizes: train=%d test=%d", len(train_dataset), len(eval_dataset))
        LOGGER.info("Batch counts: train=%d test=%d", len(train_loader), len(eval_loader))
        experiment_splits: list[tuple[int, DataLoader, DataLoader, list[int]]] = [
            (1, train_loader, eval_loader, list(range(len(eval_dataset))))
        ]
    else:
        dataset = get_dataset(config, config.dataset, train=True)
        LOGGER.info("Dataset size: train=%d", len(dataset))
        experiment_splits = []
        for fold_idx, train_indices, val_indices in build_experiment_splits(len(dataset), config):
            train_subset = Subset(dataset, train_indices.tolist())
            val_subset = Subset(dataset, val_indices.tolist())
            train_loader = create_loader(train_subset, config, shuffle=True)
            eval_loader = create_loader(val_subset, config, shuffle=False)
            experiment_splits.append((fold_idx, train_loader, eval_loader, val_indices.tolist()))
            LOGGER.info(
                "Fold %d sizes: train=%d val=%d | batches train=%d val=%d",
                fold_idx,
                len(train_subset),
                len(val_subset),
                len(train_loader),
                len(eval_loader),
            )

    for fold_idx, train_loader, eval_loader, val_indices in experiment_splits:
        model, scheduler = instantiate_model(config, device)
        model_parameter_count = count_trainable_parameters(model)
        if fold_idx == 1:
            LOGGER.info("Trainable parameters: %d", model_parameter_count)
        optimizer = Adam(model.parameters(), lr=config.lr)
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, config.epochs + 1):
            if is_diffusion(config.model):
                train_loss = train_diffusion_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device,
                    progress_label=f"[{config.dataset.upper()} | {config.model.upper()} | Fold {fold_idx}]",
                    progress_interval=config.diffusion_log_interval,
                )
                val_loss = eval_diffusion_epoch(
                    model,
                    eval_loader,
                    scheduler,
                    device,
                    progress_label=f"[{config.dataset.upper()} | {config.model.upper()} | Fold {fold_idx}]",
                    progress_interval=config.diffusion_log_interval,
                    eval_split_name="Test",
                )
                split_name = "test_loss"
            else:
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    CRITERION,
                    device,
                    model_type=config.model,
                    noise_level=noise_level,
                )
                val_loss = eval_epoch(
                    model,
                    eval_loader,
                    CRITERION,
                    device,
                    model_type=config.model,
                    noise_level=noise_level,
                )
                split_name = "val_loss"

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epoch_payload = {
                "event": "epoch_end",
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": train_loss,
                split_name: val_loss,
            }
            append_metrics_jsonl(metrics_jsonl_path, epoch_payload)
            LOGGER.info(
                "Fold %d | Epoch %d/%d | Train Loss=%.6f | %s=%.6f",
                fold_idx,
                epoch,
                config.epochs,
                train_loss,
                "Test Loss" if is_diffusion(config.model) else "Val Loss",
                val_loss,
            )

        if is_diffusion(config.model):
            metrics = evaluate_diffusion_metrics(model, eval_loader, scheduler, device)
            metrics["noise_mse"] = val_losses[-1]
        else:
            metrics = evaluate_metrics(model, eval_loader, CRITERION, device, model_type=config.model)

        fold_metrics.append(metrics)
        summary_label = "Noise MSE" if is_diffusion(config.model) else "MSE"
        summary_value = metrics["noise_mse"] if is_diffusion(config.model) else metrics["mse"]
        LOGGER.info(
            "Fold %d summary | %s=%.6f | PSNR=%s | SSIM=%s",
            fold_idx,
            summary_label,
            summary_value,
            format_metric(metrics["psnr"]),
            format_metric(metrics["ssim"]),
        )

        comparison_mse = metrics["noise_mse"] if is_diffusion(config.model) else metrics["mse"]
        best_comparison = None if best_payload is None else (
            best_payload["metrics"].get("noise_mse", best_payload["metrics"]["mse"])
        )
        if best_payload is None or comparison_mse < best_comparison:
            best_payload = {
                "fold": fold_idx,
                "metrics": metrics,
                "val_indices": val_indices,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            best_model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_payload is None or best_model_state is None:
        raise RuntimeError("Training did not produce any results.")

    all_losses = [entry.get("noise_mse", entry["mse"]) for entry in fold_metrics]
    all_psnr = [entry["psnr"] for entry in fold_metrics]
    all_ssim = [entry["ssim"] for entry in fold_metrics]
    final_summary = {
        "dataset": config.dataset,
        "model": config.model,
        "run_name": run_paths["root"].name,
        "mean_loss": float(np.mean(all_losses)),
        "std_loss": float(np.std(all_losses)),
        "psnr": mean_metric(all_psnr),
        "ssim": mean_metric(all_ssim),
        "seed": config.seed,
        "model_parameters": model_parameter_count,
        "device": str(device),
        "hostname": socket.gethostname(),
        "run_dir": str(run_paths["root"].resolve()),
        "fold_metrics": fold_metrics,
        "best_fold": best_payload,
        "artifacts": {
            "config": str(config_path.resolve()),
            "log": str((run_paths["root"] / "train.log").resolve()),
            "metrics_jsonl": str(metrics_jsonl_path.resolve()),
        },
    }

    best_model, best_scheduler = instantiate_model(config, device)
    best_model.load_state_dict(best_model_state)

    checkpoint_path = run_paths["checkpoints"] / "best.pt"
    save_checkpoint(checkpoint_path, best_model, config, final_summary)
    final_summary["artifacts"]["checkpoint"] = str(checkpoint_path.resolve())

    if is_diffusion(config.model):
        eval_dataset = get_dataset(config, config.dataset, train=False)
        best_eval_loader = create_loader(eval_dataset, config, shuffle=False)
        loss_curve_path = run_paths["plots"] / "loss_curve.png"
        plot_loss_curves(
            best_payload["train_losses"],
            best_payload["val_losses"],
            loss_curve_path,
            title=f"Diffusion Training Loss ({config.dataset.upper()}, Channels={config.base_channels})",
            y_label="MSE Loss",
        )
        sample_path = run_paths["samples"] / "generated_samples.png"
        generated_samples = sample_images(
            best_model,
            best_scheduler,
            device,
            num_samples=config.sample_count,
        )
        plot_image_grid(
            generated_samples,
            sample_path,
            title=f"Diffusion Samples ({config.dataset.upper()}, Channels={config.base_channels})",
        )
        snapshot_path = run_paths["plots"] / "diffusion_snapshots.png"
        plot_diffusion_snapshots(
            best_model,
            best_scheduler,
            device,
            dataset_name=config.dataset,
            base_channels=config.base_channels,
            save_path=snapshot_path,
            num_samples=min(config.sample_count, 8),
        )
        final_summary["artifacts"]["plots"] = [str(loss_curve_path.resolve()), str(snapshot_path.resolve())]
        final_summary["artifacts"]["samples"] = [str(sample_path.resolve())]
    else:
        dataset = get_dataset(config, config.dataset, train=True)
        best_val_subset = Subset(dataset, best_payload["val_indices"])
        best_val_loader = create_loader(best_val_subset, config, shuffle=False)
        loss_curve_path = run_paths["plots"] / "loss_curve.png"
        recon_path = run_paths["plots"] / "reconstructions.png"
        latent_path = run_paths["plots"] / "latent_space.png"

        plot_loss_curves(best_payload["train_losses"], best_payload["val_losses"], loss_curve_path)
        show_reconstructions(
            best_model,
            best_val_loader,
            device,
            recon_path,
            model_type=config.model,
            noise_level=noise_level,
        )
        plot_latent_space(best_model, best_val_loader, device, latent_path)

        artifact_plots = [str(loss_curve_path.resolve()), str(recon_path.resolve()), str(latent_path.resolve())]
        sample_artifacts: list[str] = []
        if is_vae_model(config.model):
            generated_path = run_paths["samples"] / "generated_samples.png"
            interpolation_path = run_paths["samples"] / "latent_interpolation.png"
            generate_samples(
                best_model,
                config.latent_dim,
                device,
                save_path=generated_path,
                num_samples=config.sample_count,
            )
            interpolate_images(best_model, best_val_loader, device, save_path=interpolation_path)
            sample_artifacts.extend([str(generated_path.resolve()), str(interpolation_path.resolve())])

        final_summary["artifacts"]["plots"] = artifact_plots
        final_summary["artifacts"]["samples"] = sample_artifacts

    metrics_json_path = run_paths["root"] / "metrics.json"
    save_metrics_json(metrics_json_path, final_summary)
    final_summary["artifacts"]["metrics_json"] = str(metrics_json_path.resolve())
    summary_csv_path = run_paths["root"] / "kfold_results.csv"
    save_run_summary_csv(summary_csv_path, final_summary)
    final_summary["artifacts"]["summary_csv"] = str(summary_csv_path.resolve())
    save_metrics_json(metrics_json_path, final_summary)

    LOGGER.info(
        "Final summary | dataset=%s model=%s %s=%.6f +/- %.6f | PSNR=%s | SSIM=%s",
        config.dataset,
        config.model,
        "mean_noise_loss" if is_diffusion(config.model) else "mean_loss",
        final_summary["mean_loss"],
        final_summary["std_loss"],
        format_metric(final_summary["psnr"]),
        format_metric(final_summary["ssim"]),
    )
    LOGGER.info("Artifacts saved to %s", run_paths["root"].resolve())
    return run_paths["root"]


def main() -> None:
    args = parse_args()
    cli_args = sys.argv[1:]
    selected_models = resolve_selected_models(args)
    selected_datasets = resolve_selected_datasets(args)
    base_config = build_base_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs: list[Path] = []
    for dataset_name in selected_datasets:
        for model_name in selected_models:
            run_config = build_run_config(base_config, dataset_name, model_name)
            run_dirs.append(run_single_experiment(run_config, cli_args, device))

    if run_dirs:
        print("Completed run directories:")
        for run_dir in run_dirs:
            print(run_dir.resolve())


if __name__ == "__main__":
    main()
