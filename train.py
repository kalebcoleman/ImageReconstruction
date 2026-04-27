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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "image_reconstruction_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
import numpy as np
from PIL import Image
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

from diffusion.backbones.adm_unet import (
    ADMUNet,
    default_attention_resolutions,
    default_channel_mults,
)
from diffusion.data import (
    SUPPORTED_DATASET_CHOICES,
    build_dataset,
    describe_diffusion_preprocessing,
    normalize_dataset_name as canonicalize_dataset_name,
    resolve_dataset_spec,
    resolve_diffusion_data_config,
)
from diffusion.ema import create_ema_model, select_eval_model
from diffusion.model import DiffusionUNet
from diffusion.recipes import apply_recipe_to_namespace
from diffusion.reporting import save_manifest_bundle, save_yaml
from diffusion.runtime import create_grad_scaler, format_resolved_amp_dtype
from diffusion.sampling import sample_images
from diffusion.scheduler import get_noise_schedule, predict_x0_from_model_output, q_sample
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
DATASET_CACHE: dict[tuple[str, bool, str, bool, str], Dataset] = {}
SUPPORTED_MODELS = ("ae", "dae", "vae", "diffusion")
DEFAULT_IMAGE_INTERPOLATION = "nearest"
AUTO_IMAGE_INTERPOLATION = "auto"
LOW_RES_PRESENTATION_MAX_SIZE = 32


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "all"
    dataset: str = "mnist"
    config_name: str | None = None
    config_path: Path | None = None
    protocol_name: str | None = None
    dataset_variant: str | None = None
    protocol_locked_fields: tuple[str, ...] | None = None
    protocol_allowed_overrides: tuple[str, ...] | None = None
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
    diffusion_backbone: str = "adm"
    diffusion_preprocessing: str = "default"
    image_size: int | None = None
    diffusion_channels: int | None = None
    dataset_num_classes: int | None = None
    timesteps: int = 10
    base_channels: int = 8
    time_dim: int = 64
    schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ema_decay: float = 0.0
    num_res_blocks: int = 1
    class_dropout_prob: float = 0.1
    guidance_scale: float = 1.0
    prediction_type: str = "eps"
    attention_resolutions: tuple[int, ...] | None = None
    sampler: str = "ddpm"
    sampling_steps: int | None = None
    ddim_eta: float = 0.0
    grad_clip_norm: float | None = 1.0
    amp_dtype: str = "auto"
    eval_batch_size: int | None = None
    eval_num_generated_samples: int | None = None
    eval_cfg_comparison_scales: tuple[float, ...] | None = None
    protocol_metadata: dict[str, Any] = field(default_factory=dict)
    sample_count: int = 10
    n_splits: int = 2
    dae_noise_level: float = 0.2
    diffusion_log_interval: int = 250


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train image reconstruction models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML experiment recipe. Recipe values apply first; explicit CLI flags still override them.",
    )
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
        choices=SUPPORTED_DATASET_CHOICES,
        default="mnist",
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASET_CHOICES,
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
        "--diffusion_backbone",
        "--diffusion-backbone",
        dest="diffusion_backbone",
        choices=("adm", "legacy"),
        default=ExperimentConfig.diffusion_backbone,
        help="Diffusion backbone to use. 'adm' is the scalable default; 'legacy' preserves the old MNIST-style UNet.",
    )
    parser.add_argument(
        "--legacy_diffusion",
        "--legacy-diffusion",
        dest="diffusion_backbone",
        action="store_const",
        const="legacy",
        help="Compatibility alias that selects the old MNIST-style diffusion backend.",
    )
    parser.add_argument(
        "--image_size",
        "--image-size",
        dest="image_size",
        type=int,
        default=ExperimentConfig.image_size,
        help="Diffusion image size. Defaults to 64 for ADM and native dataset size for the legacy backend.",
    )
    parser.add_argument(
        "--diffusion_channels",
        "--diffusion-channels",
        "--image_channels",
        "--image-channels",
        dest="diffusion_channels",
        type=int,
        default=ExperimentConfig.diffusion_channels,
        help="Diffusion channel count. Defaults to 3 for ADM and native dataset channels for the legacy backend.",
    )
    parser.add_argument(
        "--diffusion-preprocessing",
        dest="diffusion_preprocessing",
        choices=("default", "parity_64"),
        default=ExperimentConfig.diffusion_preprocessing,
        help="Dataset transform protocol for diffusion. 'default' preserves the current dataset-specific behavior; 'parity_64' locks the fair-comparison protocol.",
    )
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
    parser.add_argument(
        "--schedule",
        choices=("linear", "cosine"),
        default=ExperimentConfig.schedule,
        help="Diffusion beta schedule. Linear preserves the historical baseline.",
    )
    parser.add_argument("--beta_start", "--beta-start", dest="beta_start", type=float, default=ExperimentConfig.beta_start)
    parser.add_argument("--beta_end", "--beta-end", dest="beta_end", type=float, default=ExperimentConfig.beta_end)
    parser.add_argument(
        "--ema_decay",
        "--ema-decay",
        dest="ema_decay",
        type=float,
        default=ExperimentConfig.ema_decay,
        help="EMA decay for diffusion weights. Use 0 to disable EMA.",
    )
    parser.add_argument(
        "--num_res_blocks",
        "--num-res-blocks",
        dest="num_res_blocks",
        type=int,
        default=ExperimentConfig.num_res_blocks,
        help="Residual blocks per UNet stage for diffusion.",
    )
    parser.add_argument(
        "--prediction_type",
        "--prediction-type",
        "--pred-target",
        dest="prediction_type",
        choices=("eps", "v"),
        default=ExperimentConfig.prediction_type,
        help="Diffusion training target. 'eps' preserves the historical behavior; 'v' enables the improved parameterization.",
    )
    parser.add_argument(
        "--attention_resolutions",
        "--attention-resolutions",
        dest="attention_resolutions",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of spatial resolutions that should use self-attention, for example '--attention-resolutions 16 8'. Pass the flag with no values to disable attention.",
    )
    parser.add_argument(
        "--class_dropout_prob",
        "--class-dropout-prob",
        dest="class_dropout_prob",
        type=float,
        default=ExperimentConfig.class_dropout_prob,
        help="Classifier-free guidance label dropout probability for the ADM diffusion backend.",
    )
    parser.add_argument(
        "--guidance_scale",
        "--guidance-scale",
        dest="guidance_scale",
        type=float,
        default=ExperimentConfig.guidance_scale,
        help="Classifier-free guidance scale used for saved diffusion sample artifacts.",
    )
    parser.add_argument(
        "--sampler",
        "--sampling-method",
        dest="sampler",
        choices=("ddpm", "ddim"),
        default=ExperimentConfig.sampler,
        help="Sampler used for saved diffusion samples and reverse-process snapshots.",
    )
    parser.add_argument(
        "--sampling_steps",
        "--sampling-steps",
        dest="sampling_steps",
        type=int,
        default=ExperimentConfig.sampling_steps,
        help="Optional number of sampling steps. Omit for the sampler default. DDPM currently uses the full training schedule.",
    )
    parser.add_argument(
        "--ddim_eta",
        "--ddim-eta",
        dest="ddim_eta",
        type=float,
        default=ExperimentConfig.ddim_eta,
        help="DDIM stochasticity parameter. Use 0 for deterministic DDIM.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        "--grad-clip-norm",
        dest="grad_clip_norm",
        type=float,
        default=ExperimentConfig.grad_clip_norm,
        help="Gradient clipping norm for diffusion training. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--amp_dtype",
        "--amp-dtype",
        "--mixed-precision",
        "--precision",
        dest="amp_dtype",
        choices=("auto", "none", "bf16", "fp16"),
        default=ExperimentConfig.amp_dtype,
        help="Automatic mixed precision mode for diffusion training and sampling.",
    )
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
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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
    if args.image_size is not None and args.image_size < 8:
        parser.error("--image_size must be at least 8")
    if args.diffusion_channels is not None and args.diffusion_channels < 1:
        parser.error("--diffusion_channels must be at least 1")
    if args.beta_start <= 0 or args.beta_end <= 0 or args.beta_end <= args.beta_start:
        parser.error("--beta_end must be greater than --beta_start, and both must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema_decay must be in [0, 1)")
    if args.num_res_blocks < 1:
        parser.error("--num_res_blocks must be at least 1")
    if args.attention_resolutions is not None and any(resolution <= 0 for resolution in args.attention_resolutions):
        parser.error("--attention_resolutions values must all be positive")
    if not 0.0 <= args.class_dropout_prob < 1.0:
        parser.error("--class_dropout_prob must be in [0, 1)")
    if args.guidance_scale < 0.0:
        parser.error("--guidance_scale must be non-negative")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("--sampling_steps must be at least 1")
    if args.ddim_eta < 0.0:
        parser.error("--ddim_eta must be non-negative")
    if args.grad_clip_norm is not None and args.grad_clip_norm < 0.0:
        parser.error("--grad_clip_norm must be non-negative")
    if args.sample_count < 1:
        parser.error("--sample_count must be at least 1")
    if args.dae_noise_level < 0:
        parser.error("--dae_noise_level cannot be negative")
    if args.diffusion_log_interval < 0:
        parser.error("--diffusion_log_interval cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(cli_args)
    args = apply_recipe_to_namespace(args, parser=parser, argv=cli_args)
    validate_args(parser, args)
    return args


def normalize_dataset_name(dataset_name: str) -> str:
    return canonicalize_dataset_name(dataset_name)


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
        config_name=getattr(args, "config_name", None),
        config_path=getattr(args, "config_path", getattr(args, "config", None)),
        protocol_name=getattr(args, "protocol_name", None),
        dataset_variant=getattr(args, "dataset_variant", None),
        protocol_locked_fields=getattr(args, "protocol_locked_fields", None),
        protocol_allowed_overrides=getattr(args, "protocol_allowed_overrides", None),
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
        diffusion_backbone=args.diffusion_backbone,
        diffusion_preprocessing=args.diffusion_preprocessing,
        image_size=args.image_size,
        diffusion_channels=args.diffusion_channels,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        ema_decay=args.ema_decay,
        num_res_blocks=args.num_res_blocks,
        prediction_type=args.prediction_type,
        attention_resolutions=tuple(args.attention_resolutions) if args.attention_resolutions is not None else None,
        class_dropout_prob=args.class_dropout_prob,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        sampling_steps=args.sampling_steps,
        ddim_eta=args.ddim_eta,
        grad_clip_norm=args.grad_clip_norm,
        amp_dtype=args.amp_dtype,
        eval_batch_size=getattr(args, "eval_batch_size", None),
        eval_num_generated_samples=getattr(args, "eval_num_generated_samples", None),
        eval_cfg_comparison_scales=getattr(args, "eval_cfg_comparison_scales", None),
        protocol_metadata=getattr(args, "protocol_metadata", {}) or {},
        sample_count=args.sample_count,
        n_splits=args.n_splits,
        dae_noise_level=args.dae_noise_level,
        diffusion_log_interval=args.diffusion_log_interval,
    )


def build_run_config(base_config: ExperimentConfig, dataset_name: str, model_name: str) -> ExperimentConfig:
    dataset_spec = resolve_dataset_spec(dataset_name)
    resolved_image_size = dataset_spec.native_image_size
    resolved_channels = dataset_spec.native_channels
    if is_diffusion(model_name):
        resolved = resolve_diffusion_data_config(
            dataset_name,
            diffusion_backbone=base_config.diffusion_backbone,
            image_size=base_config.image_size,
            channels=base_config.diffusion_channels,
        )
        resolved_image_size = resolved.image_size
        resolved_channels = resolved.channels
    elif dataset_spec.native_channels != 1 or dataset_spec.native_image_size != 28:
        raise ValueError(
            f"{dataset_spec.name} is only supported for diffusion right now. "
            "Use --model diffusion for CIFAR10/ImageNet runs."
        )

    return ExperimentConfig(
        **{
            **asdict(base_config),
            "dataset": dataset_name,
            "model": model_name,
            "data_dir": Path(base_config.data_dir),
            "output_dir": Path(base_config.output_dir),
            "image_size": resolved_image_size,
            "diffusion_channels": resolved_channels,
            "dataset_num_classes": dataset_spec.num_classes,
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
            f"{config.dataset}_{config.model}_{config.diffusion_backbone}"
            f"_im{config.image_size}_c{config.diffusion_channels}"
            f"_{config.prediction_type}_{config.sampler}"
            f"_t{config.timesteps}_ch{config.base_channels}"
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


def resolve_legacy_artifact_paths(config: ExperimentConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)

    if is_diffusion(config.model):
        model_root = output_dir / config.model
        run_stub = slugify(config.run_name) if config.run_name else slugify(auto_run_name(config, datetime.now()))
        return {
            "loss_curve": model_root / "loss_curves" / f"{run_stub}.png",
            "samples": model_root / "samples" / f"{run_stub}.png",
            "snapshots": model_root / "snapshots" / f"{run_stub}.png",
            "reconstructions": model_root / "reconstructions" / f"{run_stub}.png",
        }

    model_root = output_dir / config.model
    run_stub = slugify(config.run_name) if config.run_name else slugify(auto_run_name(config, datetime.now()))
    return {
        "loss_curve": model_root / "loss_curves" / f"{run_stub}.png",
        "reconstructions": model_root / "reconstructions" / f"{run_stub}.png",
        "latent_space": model_root / "latent_space" / f"{run_stub}.png",
        "samples": model_root / "generated" / f"{run_stub}.png",
        "interpolations": model_root / "interpolations" / f"{run_stub}.png",
    }


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
    save_yaml(run_dir / "config.yaml", config_payload)
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
    dataset_spec = resolve_dataset_spec(dataset_name)
    if dataset_spec.name == "imagenet":
        imagenet_root = data_dir / (dataset_spec.storage_subdir or dataset_spec.name)
        return FileNotFoundError(
            "ImageNet was not found under "
            f"{imagenet_root.resolve()}. Prepare train/ and val/ subdirectories "
            "before launching diffusion jobs."
        )
    return FileNotFoundError(
        f"{dataset_name.upper()} was not found under {data_dir.resolve()}. "
        "Pre-download the dataset on a login node or rerun with --download."
    )


def get_dataset(config: ExperimentConfig, dataset_name: str, *, train: bool) -> Dataset:
    dataset_key = normalize_dataset_name(dataset_name)
    data_dir = Path(config.data_dir)
    transform_name = (
        f"diffusion_{config.image_size}_{config.diffusion_channels}_{config.diffusion_preprocessing}"
        if is_diffusion(config.model)
        else "standard"
    )
    cache_key = (dataset_key, train, str(data_dir.resolve()), config.download, transform_name)
    if cache_key in DATASET_CACHE:
        return DATASET_CACHE[cache_key]

    try:
        dataset = build_dataset(
            dataset_key,
            root=data_dir,
            train=train,
            diffusion=is_diffusion(config.model),
            image_size=config.image_size if is_diffusion(config.model) else None,
            channels=config.diffusion_channels if is_diffusion(config.model) else None,
            preprocessing_protocol=config.diffusion_preprocessing if is_diffusion(config.model) else "default",
            download=config.download,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        message = str(exc).lower()
        if not config.download and ("dataset not found" in message or "not found" in message):
            raise dataset_missing_error(dataset_key, data_dir) from exc
        if dataset_key == "imagenet" and isinstance(exc, FileNotFoundError):
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


def diffusion_to_display_range(images: torch.Tensor) -> torch.Tensor:
    """Map diffusion tensors from [-1, 1] back to [0, 1] for plotting."""

    return ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def image_for_plot(image: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a CHW tensor into a matplotlib-friendly image plus render kwargs."""

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
    """Render either grayscale or RGB tensors without hardcoding one display mode."""

    plot_tensor = image.detach().cpu().float()
    plot_image, render_kwargs = image_for_plot(plot_tensor)
    resolved_interpolation = resolve_image_interpolation(plot_tensor, interpolation=interpolation)
    if resolved_interpolation is not None:
        render_kwargs = {**render_kwargs, "interpolation": resolved_interpolation}
    axis.imshow(plot_image, **render_kwargs)


def plot_image_row(
    images: torch.Tensor,
    save_path: Path,
    title: str,
    *,
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_images = images.shape[0]
    figure, axes = plt.subplots(1, num_images, figsize=(1.5 * num_images, 2.0))
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, images):
        render_image(axis, prepare_display_images(image.unsqueeze(0))[0], interpolation=interpolation)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
    """Save a compact contact sheet without matplotlib enlargement."""

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
        if channels == 1:
            rgb_image = image.repeat(3, 1, 1)
        else:
            rgb_image = image[:3]
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
    interpolation: str | None = AUTO_IMAGE_INTERPOLATION,
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
    resolved_interpolation = resolve_image_interpolation(clean_images[0], interpolation=interpolation)
    if resolved_interpolation is not None:
        image_kwargs["interpolation"] = resolved_interpolation

    if noisy_images is not None:
        noisy_images = noisy_images.cpu()
        figure, axes = plt.subplots(3, num_images, figsize=(1.5 * num_images, 4), squeeze=False)
        for i in range(num_images):
            axes[0, i].imshow(clean_images[i].squeeze(), **image_kwargs)
            axes[1, i].imshow(noisy_images[i].squeeze(), **image_kwargs)
            axes[2, i].imshow(reconstructed_images[i].squeeze(), **image_kwargs)
            for j in range(3):
                axes[j, i].axis("off")
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Noisy")
        axes[2, 0].set_title("Reconstructed")
    else:
        figure, axes = plt.subplots(2, num_images, figsize=(1.5 * num_images, 3), squeeze=False)
        for i in range(num_images):
            axes[0, i].imshow(clean_images[i].squeeze(), **image_kwargs)
            axes[1, i].imshow(reconstructed_images[i].squeeze(), **image_kwargs)
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
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


def instantiate_model(config: ExperimentConfig, device: torch.device) -> tuple[nn.Module, Any | None]:
    scheduler = None
    if is_vae_model(config.model):
        model = VariationalAutoencoder(latent_dim=config.latent_dim).to(device)
    elif config.model in ("ae", "dae"):
        model = FullyConnectedAutoencoder(latent_dim=config.latent_dim).to(device)
    elif is_diffusion(config.model):
        if config.image_size is None or config.diffusion_channels is None:
            raise ValueError("Diffusion config must resolve image_size and diffusion_channels before instantiation.")
        if config.diffusion_backbone == "legacy":
            model = DiffusionUNet(
                in_channels=config.diffusion_channels,
                base_channels=config.base_channels,
                time_dim=config.time_dim,
                num_res_blocks=config.num_res_blocks,
            ).to(device)
        elif config.diffusion_backbone == "adm":
            attention_resolutions = (
                config.attention_resolutions
                if config.attention_resolutions is not None
                else default_attention_resolutions(config.image_size, config.dataset)
            )
            model = ADMUNet(
                in_channels=config.diffusion_channels,
                image_size=config.image_size,
                base_channels=config.base_channels,
                time_dim=config.time_dim,
                num_res_blocks=config.num_res_blocks,
                channel_mult=default_channel_mults(config.image_size),
                attention_resolutions=attention_resolutions,
                num_classes=config.dataset_num_classes,
                class_dropout_prob=config.class_dropout_prob,
            ).to(device)
        else:  # pragma: no cover - guarded by argparse.
            raise ValueError(f"Unsupported diffusion backbone: {config.diffusion_backbone}")
        scheduler = get_noise_schedule(
            config.timesteps,
            device,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule_name=config.schedule,
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
        attention_resolutions = (
            config.attention_resolutions
            if config.attention_resolutions is not None
            else default_attention_resolutions(config.image_size, config.dataset)
        )
        LOGGER.info(
            "Model hyperparameters: backbone=%s image_size=%s channels=%s classes=%s "
            "base_channels=%d time_dim=%d num_res_blocks=%d timesteps=%d "
            "schedule=%s prediction_type=%s sampler=%s sampling_steps=%s ddim_eta=%g "
            "attention_resolutions=%s beta_start=%g beta_end=%g ema_decay=%g class_dropout_prob=%g guidance_scale=%g "
            "grad_clip_norm=%s amp=%s sample_count=%d",
            config.diffusion_backbone,
            config.image_size,
            config.diffusion_channels,
            config.dataset_num_classes,
            config.base_channels,
            config.time_dim,
            config.num_res_blocks,
            config.timesteps,
            config.schedule,
            config.prediction_type,
            config.sampler,
            config.sampling_steps,
            config.ddim_eta,
            attention_resolutions,
            config.beta_start,
            config.beta_end,
            config.ema_decay,
            config.class_dropout_prob,
            config.guidance_scale,
            config.grad_clip_norm,
            format_resolved_amp_dtype(config.amp_dtype, device),
            config.sample_count,
        )
    else:
        LOGGER.info(
            "Model hyperparameters: latent_dim=%d dae_noise_level=%g n_splits=%d",
            config.latent_dim,
            config.dae_noise_level,
            config.n_splits,
        )


def save_checkpoint(
    path: Path,
    model_state_dict: dict[str, torch.Tensor],
    config: ExperimentConfig,
    metrics: dict[str, Any],
    *,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    evaluation_weights: str = "model",
) -> None:
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "ema_state_dict": ema_state_dict,
            "evaluation_weights": evaluation_weights,
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


def build_diffusion_sample_labels(
    config: ExperimentConfig,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Create a simple class-balanced label batch for diffusion sample previews."""

    if (
        not is_diffusion(config.model)
        or config.diffusion_backbone != "adm"
        or config.dataset_num_classes is None
        or config.dataset_num_classes < 1
    ):
        return None
    return torch.arange(num_samples, device=device, dtype=torch.long) % config.dataset_num_classes


def log_artifact_saved(label: str, path: Path) -> str:
    resolved_path = str(path.resolve())
    LOGGER.info("%s saved to %s", label, resolved_path)
    return resolved_path


def save_and_log_artifact(
    label: str,
    primary_path: Path,
    save_fn: Any,
    *,
    legacy_path: Path | None = None,
) -> list[str]:
    saved_paths: list[str] = []
    save_fn(primary_path)
    saved_paths.append(log_artifact_saved(label, primary_path))

    if legacy_path is not None and legacy_path.resolve() != primary_path.resolve():
        save_fn(legacy_path)
        saved_paths.append(log_artifact_saved(f"{label} (legacy path)", legacy_path))

    return saved_paths


def run_single_experiment(config: ExperimentConfig, cli_args: list[str], device: torch.device) -> Path:
    timestamp = datetime.now()
    run_paths = resolve_run_dir(config, timestamp=timestamp)
    setup_logging(run_paths["root"])
    seed_everything(config.seed)
    config_path = save_config(config, run_paths["root"], cli_args=cli_args, resolved_paths=run_paths)
    metrics_jsonl_path = run_paths["root"] / "metrics.jsonl"
    legacy_paths = resolve_legacy_artifact_paths(config)

    log_run_header(config, run_paths, device, cli_args)
    LOGGER.info("Saved resolved config to %s", config_path.resolve())

    noise_level = config.dae_noise_level if is_denoising(config.model) else 0.0
    diffusion_uses_ema = is_diffusion(config.model) and config.ema_decay > 0.0

    fold_metrics: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_ema_state: dict[str, torch.Tensor] | None = None

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
        ema_model = create_ema_model(model) if diffusion_uses_ema else None
        grad_scaler = create_grad_scaler(config.amp_dtype, device) if is_diffusion(config.model) else None
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
                    prediction_type=config.prediction_type,
                    ema_model=ema_model,
                    ema_decay=config.ema_decay,
                    amp_dtype=config.amp_dtype,
                    grad_clip_norm=config.grad_clip_norm,
                    grad_scaler=grad_scaler,
                    progress_label=f"[{config.dataset.upper()} | {config.model.upper()} | Fold {fold_idx}]",
                    progress_interval=config.diffusion_log_interval,
                )
                eval_model = select_eval_model(model, ema_model)
                val_loss = eval_diffusion_epoch(
                    eval_model,
                    eval_loader,
                    scheduler,
                    device,
                    prediction_type=config.prediction_type,
                    amp_dtype=config.amp_dtype,
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
                "evaluation_weights": "ema" if ema_model is not None else "model",
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
            eval_model = select_eval_model(model, ema_model)
            metrics = evaluate_diffusion_metrics(
                eval_model,
                eval_loader,
                scheduler,
                device,
                prediction_type=config.prediction_type,
                amp_dtype=config.amp_dtype,
            )
            metrics["noise_mse"] = val_losses[-1]
            metrics["evaluation_weights"] = "ema" if ema_model is not None else "model"
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
            best_ema_state = (
                {key: value.detach().cpu() for key, value in ema_model.state_dict().items()}
                if ema_model is not None
                else None
            )

    if best_payload is None or best_model_state is None:
        raise RuntimeError("Training did not produce any results.")

    all_losses = [entry.get("noise_mse", entry["mse"]) for entry in fold_metrics]
    all_psnr = [entry["psnr"] for entry in fold_metrics]
    all_ssim = [entry["ssim"] for entry in fold_metrics]
    final_summary = {
        "dataset": config.dataset,
        "config_name": config.config_name,
        "config_path": str(config.config_path.resolve()) if config.config_path is not None else None,
        "protocol_name": config.protocol_name,
        "dataset_variant": config.dataset_variant,
        "model": config.model,
        "run_name": run_paths["root"].name,
        "diffusion_backbone": config.diffusion_backbone if is_diffusion(config.model) else None,
        "diffusion_preprocessing": config.diffusion_preprocessing if is_diffusion(config.model) else None,
        "image_size": config.image_size if is_diffusion(config.model) else None,
        "diffusion_channels": config.diffusion_channels if is_diffusion(config.model) else None,
        "dataset_num_classes": config.dataset_num_classes if is_diffusion(config.model) else None,
        "prediction_type": config.prediction_type if is_diffusion(config.model) else None,
        "sampler": config.sampler if is_diffusion(config.model) else None,
        "sampling_steps": config.sampling_steps if is_diffusion(config.model) else None,
        "ddim_eta": config.ddim_eta if is_diffusion(config.model) else None,
        "eval_batch_size": config.eval_batch_size if is_diffusion(config.model) else None,
        "eval_num_generated_samples": config.eval_num_generated_samples if is_diffusion(config.model) else None,
        "eval_cfg_comparison_scales": (
            list(config.eval_cfg_comparison_scales)
            if is_diffusion(config.model) and config.eval_cfg_comparison_scales is not None
            else None
        ),
        "attention_resolutions": (
            list(config.attention_resolutions)
            if is_diffusion(config.model) and config.attention_resolutions is not None
            else None
        ),
        "protocol_locked_fields": list(config.protocol_locked_fields) if config.protocol_locked_fields is not None else None,
        "protocol_allowed_overrides": list(config.protocol_allowed_overrides) if config.protocol_allowed_overrides is not None else None,
        "protocol_metadata": json_ready(config.protocol_metadata) if config.protocol_metadata else {},
        "preprocessing_description": (
            describe_diffusion_preprocessing(
                config.dataset,
                image_size=config.image_size,
                channels=config.diffusion_channels,
                preprocessing_protocol=config.diffusion_preprocessing,
            )
            if is_diffusion(config.model)
            else None
        ),
        "grad_clip_norm": config.grad_clip_norm if is_diffusion(config.model) else None,
        "amp_dtype": format_resolved_amp_dtype(config.amp_dtype, device) if is_diffusion(config.model) else None,
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
        "evaluation_weights": "ema" if best_ema_state is not None else "model",
        "artifacts": {
            "config": str(config_path.resolve()),
            "config_yaml": str((run_paths["root"] / "config.yaml").resolve()),
            "log": str((run_paths["root"] / "train.log").resolve()),
            "metrics_jsonl": str(metrics_jsonl_path.resolve()),
        },
    }

    best_model, best_scheduler = instantiate_model(config, device)
    best_model.load_state_dict(best_model_state)
    best_eval_model = best_model
    if best_ema_state is not None:
        best_eval_model = create_ema_model(best_model).to(device)
        best_eval_model.load_state_dict(best_ema_state)

    checkpoint_path = run_paths["checkpoints"] / "best.pt"
    save_checkpoint(
        checkpoint_path,
        best_model_state,
        config,
        final_summary,
        ema_state_dict=best_ema_state,
        evaluation_weights=final_summary["evaluation_weights"],
    )
    final_summary["artifacts"]["checkpoint"] = str(checkpoint_path.resolve())

    if is_diffusion(config.model):
        eval_dataset = get_dataset(config, config.dataset, train=False)
        best_eval_loader = create_loader(eval_dataset, config, shuffle=False)
        image_shape = (
            config.diffusion_channels,
            config.image_size,
            config.image_size,
        )
        sample_labels = build_diffusion_sample_labels(config, config.sample_count, device)
        loss_curve_path = run_paths["plots"] / "loss_curve.png"
        logged_loss_curve_paths = save_and_log_artifact(
            "Diffusion loss curve",
            loss_curve_path,
            lambda target: plot_loss_curves(
                best_payload["train_losses"],
                best_payload["val_losses"],
                target,
                title=(
                    f"Diffusion Training Loss ({config.dataset.upper()}, {config.diffusion_backbone}, "
                    f"{config.image_size}x{config.image_size}, C={config.diffusion_channels}, "
                    f"{config.prediction_type}, {config.sampler})"
                ),
                y_label="MSE Loss",
            ),
            legacy_path=legacy_paths["loss_curve"],
        )
        sample_path = run_paths["samples"] / "generated_samples.png"
        generated_samples = sample_images(
            best_eval_model,
            best_scheduler,
            device,
            num_samples=config.sample_count,
            image_shape=image_shape,
            labels=sample_labels,
            guidance_scale=config.guidance_scale,
            prediction_type=config.prediction_type,
            sampler_name=config.sampler,
            sampling_steps=config.sampling_steps,
            ddim_eta=config.ddim_eta,
            amp_dtype=config.amp_dtype,
        )
        logged_sample_paths = save_and_log_artifact(
            "Diffusion sample grid",
            sample_path,
            lambda target: plot_image_grid(
                generated_samples,
                target,
                title=(
                    f"Diffusion Samples ({config.dataset.upper()}, {config.diffusion_backbone}, "
                    f"{config.image_size}x{config.image_size}, C={config.diffusion_channels}, "
                    f"{config.prediction_type}, {config.sampler})"
                ),
            ),
            legacy_path=legacy_paths["samples"],
        )
        native_sample_path = run_paths["samples"] / "generated_samples_native_grid.png"
        logged_native_sample_paths = save_and_log_artifact(
            "Diffusion native sample grid",
            native_sample_path,
            lambda target: save_native_image_grid(generated_samples, target),
        )
        snapshot_path = run_paths["plots"] / "diffusion_snapshots.png"
        logged_snapshot_paths = save_and_log_artifact(
            "Diffusion reverse-process snapshots",
            snapshot_path,
            lambda target: plot_diffusion_snapshots(
                best_eval_model,
                best_scheduler,
                device,
                dataset_name=config.dataset,
                image_shape=image_shape,
                base_channels=config.base_channels,
                save_path=target,
                num_samples=min(config.sample_count, 8),
                sample_labels=(
                    sample_labels[: min(config.sample_count, 8)]
                    if sample_labels is not None
                    else None
                ),
                guidance_scale=config.guidance_scale,
                prediction_type=config.prediction_type,
                sampler_name=config.sampler,
                sampling_steps=config.sampling_steps,
                ddim_eta=config.ddim_eta,
                amp_dtype=config.amp_dtype,
            ),
            legacy_path=legacy_paths["snapshots"],
        )
        reconstruction_path = run_paths["plots"] / "reconstructions.png"
        logged_reconstruction_paths = save_and_log_artifact(
            "Diffusion reconstruction preview",
            reconstruction_path,
            lambda target: plot_diffusion_reconstructions(
                best_eval_model,
                best_scheduler,
                best_eval_loader,
                device,
                dataset_name=config.dataset,
                base_channels=config.base_channels,
                prediction_type=config.prediction_type,
                save_path=target,
                num_images=min(config.sample_count, 8),
            ),
            legacy_path=legacy_paths["reconstructions"],
        )
        final_summary["artifacts"]["plots"] = (
            logged_loss_curve_paths + logged_snapshot_paths + logged_reconstruction_paths
        )
        final_summary["artifacts"]["samples"] = logged_sample_paths + logged_native_sample_paths
    else:
        dataset = get_dataset(config, config.dataset, train=True)
        best_val_subset = Subset(dataset, best_payload["val_indices"])
        best_val_loader = create_loader(best_val_subset, config, shuffle=False)
        loss_curve_path = run_paths["plots"] / "loss_curve.png"
        recon_path = run_paths["plots"] / "reconstructions.png"
        latent_path = run_paths["plots"] / "latent_space.png"

        plot_loss_curves(best_payload["train_losses"], best_payload["val_losses"], loss_curve_path)
        logged_loss_curve_path = log_artifact_saved("Loss curve", loss_curve_path)
        show_reconstructions(
            best_model,
            best_val_loader,
            device,
            recon_path,
            model_type=config.model,
            noise_level=noise_level,
        )
        logged_recon_path = log_artifact_saved("Reconstruction preview", recon_path)
        plot_latent_space(best_model, best_val_loader, device, latent_path)
        logged_latent_path = log_artifact_saved("Latent space plot", latent_path)

        artifact_plots = [logged_loss_curve_path, logged_recon_path, logged_latent_path]
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
            logged_generated_path = log_artifact_saved("VAE generated sample grid", generated_path)
            interpolate_images(best_model, best_val_loader, device, save_path=interpolation_path)
            logged_interpolation_path = log_artifact_saved("VAE interpolation grid", interpolation_path)
            sample_artifacts.extend([logged_generated_path, logged_interpolation_path])

        final_summary["artifacts"]["plots"] = artifact_plots
        final_summary["artifacts"]["samples"] = sample_artifacts

    metrics_json_path = run_paths["root"] / "metrics.json"
    save_metrics_json(metrics_json_path, final_summary)
    final_summary["artifacts"]["metrics_json"] = str(metrics_json_path.resolve())
    summary_csv_path = run_paths["root"] / "kfold_results.csv"
    save_run_summary_csv(summary_csv_path, final_summary)
    final_summary["artifacts"]["summary_csv"] = str(summary_csv_path.resolve())
    manifest_payload = {
        "config": json_ready(asdict(config)),
        "summary": json_ready(final_summary),
    }
    manifest_paths = save_manifest_bundle(
        run_paths["root"],
        basename="run_manifest",
        title=f"Run Manifest: {run_paths['root'].name}",
        payload=manifest_payload,
    )
    final_summary["artifacts"]["run_manifest"] = manifest_paths
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
