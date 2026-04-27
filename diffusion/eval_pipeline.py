from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion.data import build_dataset, describe_diffusion_preprocessing
from diffusion.ema import create_ema_model
from diffusion.eval_metrics import (
    FeatureStatistics,
    InceptionFeatureExtractor,
    LPIPSDiversityMetric,
    RunningFeatureStats,
    compute_fid,
    compute_inception_score,
    load_feature_statistics,
    prepare_images_for_metrics,
    save_feature_statistics,
)
from diffusion.reporting import save_manifest_bundle, save_yaml
from diffusion.sampling import sample_images
from diffusion.training import evaluate_diffusion_metrics
from train import (
    ExperimentConfig,
    instantiate_model,
    json_ready,
    plot_diffusion_reconstructions,
    plot_diffusion_snapshots,
    plot_image_grid,
    prepare_display_images,
    render_image,
    save_native_image_grid,
)


@dataclass(frozen=True)
class CheckpointEvaluationConfig:
    """Runtime settings for checkpoint-only evaluation and sampling."""

    checkpoint_path: Path
    mode: str = "evaluate"
    output_dir: Path | None = None
    run_name: str | None = None
    data_dir: Path | None = None
    num_generated_samples: int | None = None
    eval_batch_size: int | None = None
    num_workers: int = 0
    sampler: str | None = None
    sampling_steps: int | None = None
    ddim_eta: float | None = None
    guidance_scale: float | None = None
    amp_dtype: str | None = None
    artifact_sample_count: int | None = None
    cfg_comparison_scales: tuple[float, ...] | None = None
    save_raw_images: bool = False
    paired_metrics: bool = True
    reference_stats_dir: Path | None = None
    allow_model_download: bool = False
    nearest_neighbor_count: int = 8
    nearest_neighbor_reference_limit: int = 10_000
    force_unconditional: bool = False
    lpips_pair_count: int = 128


class MetricBackend:
    """Container for the pretrained models used by the evaluation stack."""

    def __init__(self, device: torch.device, *, allow_model_download: bool) -> None:
        self.inception = InceptionFeatureExtractor(
            device,
            allow_model_download=allow_model_download,
        )
        self.lpips = LPIPSDiversityMetric(
            device,
            allow_model_download=allow_model_download,
        )


def create_metric_backend(
    device: torch.device,
    *,
    allow_model_download: bool,
) -> MetricBackend:
    """Factory wrapper so tests can inject a cheaper backend."""

    return MetricBackend(device, allow_model_download=allow_model_download)


def _config_from_checkpoint_payload(payload: dict[str, Any]) -> ExperimentConfig:
    """Reconstruct the experiment config stored inside a checkpoint."""

    config_payload = dict(payload.get("config") or {})
    valid_fields = {field.name for field in fields(ExperimentConfig)}
    filtered = {key: value for key, value in config_payload.items() if key in valid_fields}
    if "data_dir" in filtered:
        filtered["data_dir"] = Path(filtered["data_dir"])
    if "output_dir" in filtered:
        filtered["output_dir"] = Path(filtered["output_dir"])
    if "config_path" in filtered and filtered["config_path"] is not None:
        filtered["config_path"] = Path(filtered["config_path"])
    for tuple_field in (
        "attention_resolutions",
        "eval_cfg_comparison_scales",
        "protocol_locked_fields",
        "protocol_allowed_overrides",
    ):
        if isinstance(filtered.get(tuple_field), list):
            filtered[tuple_field] = tuple(filtered[tuple_field])
    return ExperimentConfig(**filtered)


def _apply_evaluation_overrides(
    checkpoint_config: ExperimentConfig,
    evaluation_config: CheckpointEvaluationConfig,
) -> ExperimentConfig:
    """Overlay checkpoint-evaluation overrides onto the saved training config."""

    updated = {
        **asdict(checkpoint_config),
        "data_dir": Path(evaluation_config.data_dir or checkpoint_config.data_dir),
        "num_workers": evaluation_config.num_workers,
    }
    if evaluation_config.sampler is not None:
        updated["sampler"] = evaluation_config.sampler
    if evaluation_config.sampling_steps is not None:
        updated["sampling_steps"] = evaluation_config.sampling_steps
    if evaluation_config.ddim_eta is not None:
        updated["ddim_eta"] = evaluation_config.ddim_eta
    if evaluation_config.guidance_scale is not None:
        updated["guidance_scale"] = evaluation_config.guidance_scale
    if evaluation_config.amp_dtype is not None:
        updated["amp_dtype"] = evaluation_config.amp_dtype
    return ExperimentConfig(
        **{
            **updated,
            "data_dir": Path(updated["data_dir"]),
            "output_dir": Path(updated["output_dir"]),
            "attention_resolutions": (
                tuple(updated["attention_resolutions"])
                if updated.get("attention_resolutions") is not None
                else None
            ),
        }
    )


def _resolve_checkpoint_run_root(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _resolve_evaluation_paths(
    checkpoint_path: Path,
    *,
    output_dir: Path | None,
    run_name: str | None,
) -> dict[str, Path]:
    run_root = _resolve_checkpoint_run_root(checkpoint_path)
    evaluation_root = Path(output_dir) if output_dir is not None else run_root / "evaluations"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    requested_name = run_name or f"{checkpoint_path.stem}_evaluation"
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in requested_name).strip("_")
    candidate = evaluation_root / safe_name
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = evaluation_root / f"{safe_name}_{suffix:02d}"

    paths = {
        "root": candidate,
        "artifacts": candidate / "artifacts",
        "generated_images": candidate / "generated_images",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _flatten_payload(prefix: str, payload: dict[str, Any], flat: dict[str, Any]) -> None:
    for key, value in payload.items():
        nested_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_payload(nested_key, value, flat)
        else:
            flat[nested_key] = value


def _save_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    flat: dict[str, Any] = {}
    _flatten_payload("", payload, flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def _build_loader(
    config: ExperimentConfig,
    *,
    train_split: bool,
    batch_size: int,
) -> DataLoader:
    dataset = build_dataset(
        config.dataset,
        root=Path(config.data_dir),
        train=train_split,
        diffusion=True,
        image_size=config.image_size,
        channels=config.diffusion_channels,
        preprocessing_protocol=config.diffusion_preprocessing,
        download=config.download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def _reference_stats_path(
    config: ExperimentConfig,
    *,
    reference_stats_dir: Path,
    backend: MetricBackend,
) -> Path:
    filename = (
        f"{config.dataset}_eval_im{config.image_size}_c{config.diffusion_channels}_"
        f"proc-{config.diffusion_preprocessing}_"
        f"{backend.inception.preprocess_signature}.npz"
    )
    return reference_stats_dir / filename


def _load_or_compute_reference_stats(
    config: ExperimentConfig,
    *,
    backend: MetricBackend,
    reference_stats_dir: Path,
    batch_size: int,
) -> tuple[FeatureStatistics, Path]:
    reference_stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = _reference_stats_path(config, reference_stats_dir=reference_stats_dir, backend=backend)
    expected_metadata = {
        "dataset": config.dataset,
        "split": "eval",
        "image_size": config.image_size,
        "channels": config.diffusion_channels,
        "diffusion_preprocessing": config.diffusion_preprocessing,
        "preprocess_signature": backend.inception.preprocess_signature,
    }

    if stats_path.exists():
        stats, metadata = load_feature_statistics(stats_path)
        if metadata == expected_metadata:
            return stats, stats_path

    loader = _build_loader(
        config,
        train_split=False,
        batch_size=batch_size,
    )
    accumulator = RunningFeatureStats(backend.inception.feature_dim)
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        features, _ = backend.inception.extract(images)
        accumulator.update(features)
    stats = accumulator.finalize()
    save_feature_statistics(stats_path, stats, expected_metadata)
    return stats, stats_path


def _balanced_labels(
    config: ExperimentConfig,
    *,
    num_samples: int,
    device: torch.device,
    start_index: int = 0,
    force_unconditional: bool = False,
) -> torch.Tensor | None:
    if force_unconditional:
        return None
    if config.dataset_num_classes is None or config.dataset_num_classes < 1:
        return None
    labels = torch.arange(start_index, start_index + num_samples, device=device, dtype=torch.long)
    return labels % config.dataset_num_classes


def _assert_generated_images(images: torch.Tensor, image_shape: tuple[int, int, int]) -> None:
    if images.ndim != 4:
        raise ValueError(f"Expected generated images with shape (N, C, H, W), got {tuple(images.shape)}.")
    if images.shape[1:] != image_shape:
        raise ValueError(f"Expected generated images with shape (*, {image_shape}), got {tuple(images.shape)}.")
    if float(images.min()) < -1e-6 or float(images.max()) > 1.0 + 1e-6:
        raise ValueError(
            f"Generated images must be in [0, 1] for evaluation, observed range "
            f"[{float(images.min())}, {float(images.max())}]."
        )


def _save_raw_images(
    images: torch.Tensor,
    *,
    start_index: int,
    output_dir: Path,
    labels: torch.Tensor | None,
) -> list[str]:
    saved_paths: list[str] = []
    for image_idx, image in enumerate(images):
        array = prepare_images_for_metrics(image.unsqueeze(0))[0]
        if array.shape[0] == 1:
            pil_image = Image.fromarray((array.squeeze(0).numpy() * 255.0).round().astype(np.uint8), mode="L")
        else:
            pil_image = Image.fromarray(
                (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
                mode="RGB",
            )
        label_suffix = ""
        if labels is not None:
            label_suffix = f"_label{int(labels[image_idx].item()):04d}"
        path = output_dir / f"{start_index + image_idx:06d}{label_suffix}.png"
        pil_image.save(path)
        saved_paths.append(str(path.resolve()))
    return saved_paths


def _save_cfg_comparison_grid(
    model: torch.nn.Module,
    scheduler: Any,
    device: torch.device,
    config: ExperimentConfig,
    *,
    save_path: Path,
    guidance_scales: tuple[float, ...],
    num_samples: int,
    force_unconditional: bool,
) -> None:
    if not guidance_scales:
        return
    image_shape = (config.diffusion_channels, config.image_size, config.image_size)
    labels = _balanced_labels(
        config,
        num_samples=num_samples,
        device=device,
        force_unconditional=force_unconditional,
    )
    initial_noise = torch.randn(num_samples, *image_shape, device=device)
    figure, axes = plt.subplots(len(guidance_scales), num_samples, figsize=(1.7 * num_samples, 1.7 * len(guidance_scales)), squeeze=False)

    for row_idx, guidance_scale in enumerate(guidance_scales):
        images = sample_images(
            model,
            scheduler,
            device,
            num_samples=num_samples,
            image_shape=image_shape,
            initial_noise=initial_noise.clone(),
            labels=labels,
            guidance_scale=guidance_scale,
            prediction_type=config.prediction_type,
            sampler_name=config.sampler,
            sampling_steps=config.sampling_steps,
            ddim_eta=config.ddim_eta,
            amp_dtype=config.amp_dtype,
        )
        display_images = prepare_display_images(images)
        for col_idx in range(num_samples):
            axis = axes[row_idx, col_idx]
            render_image(axis, display_images[col_idx])
            axis.axis("off")
            if col_idx == 0:
                axis.set_ylabel(f"cfg={guidance_scale:g}", rotation=0, labelpad=28, va="center")

    figure.suptitle("CFG Comparison", fontsize=13)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_nearest_neighbor_grid(
    generated_images: torch.Tensor,
    *,
    config: ExperimentConfig,
    save_path: Path,
    batch_size: int,
    max_reference_images: int,
) -> None:
    if generated_images.numel() == 0:
        return

    reference_loader = _build_loader(
        config,
        train_split=True,
        batch_size=batch_size,
    )
    reference_images: list[torch.Tensor] = []
    for batch in reference_loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        reference_images.append(prepare_images_for_metrics(images))
        if sum(chunk.shape[0] for chunk in reference_images) >= max_reference_images:
            break

    real_bank = torch.cat(reference_images, dim=0)[:max_reference_images]
    if real_bank.shape[0] == 0:
        raise ValueError("Nearest-neighbor retrieval requires at least one real reference image.")

    generated = prepare_images_for_metrics(generated_images)
    generated_flat = generated.reshape(generated.shape[0], -1)
    real_flat = real_bank.reshape(real_bank.shape[0], -1)
    distances = torch.cdist(generated_flat, real_flat, p=2)
    nearest_indices = distances.argmin(dim=1)

    figure, axes = plt.subplots(2, generated.shape[0], figsize=(1.8 * generated.shape[0], 3.6), squeeze=False)
    for image_idx in range(generated.shape[0]):
        generated_axis = axes[0, image_idx]
        neighbor_axis = axes[1, image_idx]
        render_image(generated_axis, generated[image_idx])
        render_image(neighbor_axis, real_bank[nearest_indices[image_idx]])
        generated_axis.axis("off")
        neighbor_axis.axis("off")
        if image_idx == 0:
            generated_axis.set_ylabel("Generated", rotation=0, labelpad=28, va="center")
            neighbor_axis.set_ylabel("Nearest", rotation=0, labelpad=28, va="center")
        neighbor_axis.set_title(f"d={float(distances[image_idx, nearest_indices[image_idx]]):.3f}")

    figure.suptitle("Nearest Real Neighbors", fontsize=13)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _collect_generated_outputs(
    model: torch.nn.Module,
    scheduler: Any,
    device: torch.device,
    config: ExperimentConfig,
    *,
    num_generated_samples: int,
    batch_size: int,
    guidance_scale: float,
    force_unconditional: bool,
    save_raw_images: bool,
    raw_output_dir: Path,
    metric_backend: MetricBackend | None,
    lpips_pair_count: int,
    artifact_buffer_limit: int,
) -> dict[str, Any]:
    image_shape = (config.diffusion_channels, config.image_size, config.image_size)
    fake_stats = RunningFeatureStats(metric_backend.inception.feature_dim) if metric_backend is not None else None
    logits_chunks: list[torch.Tensor] = []
    lpips_images: list[torch.Tensor] = []
    artifact_images: list[torch.Tensor] = []
    raw_image_paths: list[str] = []

    generated_count = 0
    while generated_count < num_generated_samples:
        current_batch = min(batch_size, num_generated_samples - generated_count)
        labels = _balanced_labels(
            config,
            num_samples=current_batch,
            device=device,
            start_index=generated_count,
            force_unconditional=force_unconditional,
        )
        generated = sample_images(
            model,
            scheduler,
            device,
            num_samples=current_batch,
            image_shape=image_shape,
            labels=labels,
            guidance_scale=guidance_scale,
            prediction_type=config.prediction_type,
            sampler_name=config.sampler,
            sampling_steps=config.sampling_steps,
            ddim_eta=config.ddim_eta,
            amp_dtype=config.amp_dtype,
        )
        _assert_generated_images(generated, image_shape)
        generated_count += current_batch

        if sum(chunk.shape[0] for chunk in artifact_images) < artifact_buffer_limit:
            remaining = artifact_buffer_limit - sum(chunk.shape[0] for chunk in artifact_images)
            artifact_images.append(generated[:remaining])

        if metric_backend is not None and fake_stats is not None:
            features, logits = metric_backend.inception.extract(generated)
            fake_stats.update(features)
            logits_chunks.append(logits)
            if sum(chunk.shape[0] for chunk in lpips_images) < lpips_pair_count * 2:
                lpips_images.append(generated)

        if save_raw_images:
            raw_image_paths.extend(
                _save_raw_images(
                    generated,
                    start_index=generated_count - current_batch,
                    output_dir=raw_output_dir,
                    labels=labels,
                )
            )

    artifact_tensor = torch.cat(artifact_images, dim=0) if artifact_images else torch.empty(0, *image_shape)
    return {
        "fake_stats": fake_stats.finalize() if fake_stats is not None else None,
        "logits": torch.cat(logits_chunks, dim=0) if logits_chunks else None,
        "artifact_images": artifact_tensor,
        "lpips_images": torch.cat(lpips_images, dim=0) if lpips_images else None,
        "raw_image_paths": raw_image_paths,
    }


def run_checkpoint_evaluation(
    evaluation_config: CheckpointEvaluationConfig,
    *,
    device: torch.device | None = None,
    metric_backend_factory: Callable[..., MetricBackend] = create_metric_backend,
) -> dict[str, Any]:
    """Evaluate or sample from a diffusion checkpoint without retraining."""

    checkpoint_path = Path(evaluation_config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = _config_from_checkpoint_payload(checkpoint_payload)
    if checkpoint_config.model != "diffusion":
        raise ValueError("Checkpoint evaluation currently supports diffusion checkpoints only.")

    config = _apply_evaluation_overrides(checkpoint_config, evaluation_config)
    effective_eval_batch_size = evaluation_config.eval_batch_size or config.eval_batch_size or 32
    effective_num_generated = (
        evaluation_config.num_generated_samples
        or config.eval_num_generated_samples
        or 1000
    )
    effective_artifact_sample_count = evaluation_config.artifact_sample_count or config.sample_count or 16
    paths = _resolve_evaluation_paths(
        checkpoint_path,
        output_dir=evaluation_config.output_dir,
        run_name=evaluation_config.run_name,
    )

    model, scheduler = instantiate_model(config, resolved_device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    evaluation_weights = checkpoint_payload.get("evaluation_weights", "model")
    eval_model = model
    ema_state = checkpoint_payload.get("ema_state_dict")
    if evaluation_weights == "ema" and ema_state is not None:
        eval_model = create_ema_model(model).to(resolved_device)
        eval_model.load_state_dict(ema_state)
    eval_model.eval()

    effective_num_generated = max(effective_num_generated, effective_artifact_sample_count)
    save_raw_images = evaluation_config.save_raw_images
    metric_backend = None
    reference_stats = None
    reference_stats_path = None
    if evaluation_config.mode == "evaluate":
        metric_backend = metric_backend_factory(
            resolved_device,
            allow_model_download=evaluation_config.allow_model_download,
        )
        reference_stats_dir = Path(
            evaluation_config.reference_stats_dir
            or (Path(config.output_dir) / "_reference_stats")
        )
        reference_stats, reference_stats_path = _load_or_compute_reference_stats(
            config,
            backend=metric_backend,
            reference_stats_dir=reference_stats_dir,
            batch_size=effective_eval_batch_size,
        )

    generated_payload = _collect_generated_outputs(
        eval_model,
        scheduler,
        resolved_device,
        config,
        num_generated_samples=effective_num_generated,
        batch_size=effective_eval_batch_size,
        guidance_scale=config.guidance_scale,
        force_unconditional=evaluation_config.force_unconditional,
        save_raw_images=save_raw_images,
        raw_output_dir=paths["generated_images"],
        metric_backend=metric_backend,
        lpips_pair_count=evaluation_config.lpips_pair_count,
        artifact_buffer_limit=max(
            effective_artifact_sample_count,
            evaluation_config.lpips_pair_count * 2 if metric_backend is not None else 0,
        ),
    )
    artifact_images = generated_payload["artifact_images"][: effective_artifact_sample_count]

    artifact_paths: dict[str, Any] = {}
    sample_grid_path = paths["artifacts"] / "generated_samples.png"
    plot_image_grid(
        artifact_images,
        sample_grid_path,
        title=f"Generated Samples ({config.dataset.upper()}, {config.sampler}, cfg={config.guidance_scale:g})",
    )
    artifact_paths["generated_sample_grid"] = str(sample_grid_path.resolve())
    artifact_paths["generated_samples"] = str(sample_grid_path.resolve())
    native_sample_grid_path = paths["artifacts"] / "generated_samples_native_grid.png"
    save_native_image_grid(artifact_images, native_sample_grid_path)
    artifact_paths["generated_native_grid"] = str(native_sample_grid_path.resolve())
    artifact_paths["generated_samples_native_grid"] = str(native_sample_grid_path.resolve())

    if config.dataset_num_classes is not None and not evaluation_config.force_unconditional:
        class_grid_labels = _balanced_labels(
            config,
            num_samples=effective_artifact_sample_count,
            device=resolved_device,
        )
        class_grid_images = sample_images(
            eval_model,
            scheduler,
            resolved_device,
            num_samples=effective_artifact_sample_count,
            image_shape=(config.diffusion_channels, config.image_size, config.image_size),
            labels=class_grid_labels,
            guidance_scale=config.guidance_scale,
            prediction_type=config.prediction_type,
            sampler_name=config.sampler,
            sampling_steps=config.sampling_steps,
            ddim_eta=config.ddim_eta,
            amp_dtype=config.amp_dtype,
        )
        class_grid_path = paths["artifacts"] / "class_conditional_samples.png"
        plot_image_grid(
            class_grid_images,
            class_grid_path,
            title="Class-Conditional Samples",
        )
        artifact_paths["class_conditional_sample_grid"] = str(class_grid_path.resolve())

    cfg_scales = evaluation_config.cfg_comparison_scales
    if cfg_scales is None:
        cfg_scales = (
            config.eval_cfg_comparison_scales
            if config.eval_cfg_comparison_scales is not None
            else tuple(sorted({0.0, 1.0, float(config.guidance_scale)}))
        )
    if cfg_scales and config.dataset_num_classes is not None and not evaluation_config.force_unconditional:
        cfg_grid_path = paths["artifacts"] / "cfg_comparison.png"
        _save_cfg_comparison_grid(
            eval_model,
            scheduler,
            resolved_device,
            config,
            save_path=cfg_grid_path,
            guidance_scales=cfg_scales,
            num_samples=min(4, effective_artifact_sample_count),
            force_unconditional=evaluation_config.force_unconditional,
        )
        artifact_paths["cfg_comparison_grid"] = str(cfg_grid_path.resolve())

    snapshot_path = paths["artifacts"] / "diffusion_snapshots.png"
    plot_diffusion_snapshots(
        eval_model,
        scheduler,
        resolved_device,
        dataset_name=config.dataset,
        image_shape=(config.diffusion_channels, config.image_size, config.image_size),
        base_channels=config.base_channels,
        save_path=snapshot_path,
        num_samples=min(4, effective_artifact_sample_count),
        sample_labels=_balanced_labels(
            config,
            num_samples=min(4, effective_artifact_sample_count),
            device=resolved_device,
            force_unconditional=evaluation_config.force_unconditional,
        ),
        guidance_scale=config.guidance_scale,
        prediction_type=config.prediction_type,
        sampler_name=config.sampler,
        sampling_steps=config.sampling_steps,
        ddim_eta=config.ddim_eta,
        amp_dtype=config.amp_dtype,
    )
    artifact_paths["reverse_process_snapshots"] = str(snapshot_path.resolve())
    artifact_paths["diffusion_snapshots"] = str(snapshot_path.resolve())

    if evaluation_config.mode == "evaluate":
        nearest_neighbor_path = paths["artifacts"] / "nearest_neighbors.png"
        _save_nearest_neighbor_grid(
            artifact_images[: min(evaluation_config.nearest_neighbor_count, artifact_images.shape[0])],
            config=config,
            save_path=nearest_neighbor_path,
            batch_size=effective_eval_batch_size,
            max_reference_images=evaluation_config.nearest_neighbor_reference_limit,
        )
        artifact_paths["nearest_neighbor_grid"] = str(nearest_neighbor_path.resolve())

    paired_metrics_payload: dict[str, Any] | None = None
    if evaluation_config.mode == "evaluate" and evaluation_config.paired_metrics:
        eval_loader = _build_loader(
            config,
            train_split=False,
            batch_size=effective_eval_batch_size,
        )
        paired_metrics = evaluate_diffusion_metrics(
            eval_model,
            eval_loader,
            scheduler,
            resolved_device,
            prediction_type=config.prediction_type,
            amp_dtype=config.amp_dtype,
        )
        paired_metrics_payload = {
            "description": "Auxiliary paired denoising / reconstruction metrics. These are not primary generative metrics.",
            **paired_metrics,
        }
        reconstruction_path = paths["artifacts"] / "reconstructions.png"
        plot_diffusion_reconstructions(
            eval_model,
            scheduler,
            eval_loader,
            resolved_device,
            dataset_name=config.dataset,
            base_channels=config.base_channels,
            prediction_type=config.prediction_type,
            save_path=reconstruction_path,
            num_images=min(8, effective_artifact_sample_count),
        )
        artifact_paths["reconstruction_preview"] = str(reconstruction_path.resolve())
        artifact_paths["reconstructions"] = str(reconstruction_path.resolve())

    generative_metrics_payload: dict[str, Any] | None = None
    if evaluation_config.mode == "evaluate" and metric_backend is not None and reference_stats is not None:
        fake_stats = generated_payload["fake_stats"]
        fake_logits = generated_payload["logits"]
        lpips_images = generated_payload["lpips_images"]
        if fake_stats is None or fake_logits is None:
            raise RuntimeError("Generative evaluation requires fake statistics and logits.")

        fid_value = compute_fid(reference_stats, fake_stats)
        inception_mean, inception_std = compute_inception_score(fake_logits)
        lpips_value = metric_backend.lpips.compute(lpips_images, pair_count=evaluation_config.lpips_pair_count)
        generative_metrics_payload = {
            "fid": fid_value,
            "inception_score_mean": inception_mean,
            "inception_score_std": inception_std,
            "lpips_diversity": lpips_value,
            "lpips_interpretation": metric_backend.lpips.description,
            "lpips_backend": metric_backend.lpips.backend_name,
            "primary_metric": "fid",
            "secondary_metric": "inception_score_mean",
            "num_generated_samples": effective_num_generated,
        }

    evaluation_payload = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "mode": evaluation_config.mode,
        "evaluation_weights": evaluation_weights,
        "device": str(resolved_device),
        "dataset": config.dataset,
        "config_name": config.config_name,
        "protocol_name": config.protocol_name,
        "dataset_variant": config.dataset_variant,
        "seed": config.seed,
        "protocol_locked_fields": list(config.protocol_locked_fields) if config.protocol_locked_fields is not None else None,
        "protocol_allowed_overrides": list(config.protocol_allowed_overrides) if config.protocol_allowed_overrides is not None else None,
        "protocol_metadata": json_ready(config.protocol_metadata) if config.protocol_metadata else {},
        "diffusion_backbone": config.diffusion_backbone,
        "image_size": config.image_size,
        "diffusion_channels": config.diffusion_channels,
        "diffusion_preprocessing": config.diffusion_preprocessing,
        "preprocessing_description": describe_diffusion_preprocessing(
            config.dataset,
            image_size=config.image_size,
            channels=config.diffusion_channels,
            preprocessing_protocol=config.diffusion_preprocessing,
        ),
        "prediction_type": config.prediction_type,
        "sampler": config.sampler,
        "sampling_steps": config.sampling_steps,
        "ddim_eta": config.ddim_eta,
        "guidance_scale": config.guidance_scale,
        "amp_dtype": config.amp_dtype,
        "model_parameters": checkpoint_payload.get("metrics", {}).get("model_parameters"),
        "num_generated_samples": effective_num_generated,
        "eval_batch_size": effective_eval_batch_size,
        "generative_metrics": generative_metrics_payload,
        "paired_metrics": paired_metrics_payload,
        "reference_stats_path": str(reference_stats_path.resolve()) if reference_stats_path is not None else None,
        "artifacts": artifact_paths,
        "raw_image_paths": generated_payload["raw_image_paths"] if save_raw_images else [],
        "cfg_comparison_scales": list(cfg_scales) if cfg_scales is not None else None,
    }

    config_payload = {
        "evaluation": asdict(evaluation_config),
        "checkpoint_config": json_ready(asdict(config)),
    }
    config_path = paths["root"] / "evaluation_config.json"
    _save_json(config_path, config_payload)
    metrics_path = paths["root"] / "metrics.json"
    _save_json(metrics_path, evaluation_payload)
    summary_path = paths["root"] / "metrics_summary.csv"
    _save_summary_csv(summary_path, evaluation_payload)
    evaluation_payload["evaluation_config_path"] = str(config_path.resolve())
    evaluation_payload["evaluation_config_yaml_path"] = str((paths["root"] / "evaluation_config.yaml").resolve())
    evaluation_payload["metrics_path"] = str(metrics_path.resolve())
    evaluation_payload["summary_path"] = str(summary_path.resolve())
    evaluation_payload["evaluation_dir"] = str(paths["root"].resolve())
    save_yaml(paths["root"] / "evaluation_config.yaml", config_payload)
    evaluation_manifest = {
        "checkpoint_config": json_ready(asdict(config)),
        "evaluation": json_ready(evaluation_payload),
    }
    manifest_paths = save_manifest_bundle(
        paths["root"],
        basename="evaluation_manifest",
        title=f"Evaluation Manifest: {paths['root'].name}",
        payload=evaluation_manifest,
    )
    evaluation_payload["manifest_paths"] = manifest_paths
    _save_json(metrics_path, evaluation_payload)
    return evaluation_payload
