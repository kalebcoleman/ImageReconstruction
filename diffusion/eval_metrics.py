from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import AlexNet_Weights, Inception_V3_Weights


def prepare_images_for_metrics(images: torch.Tensor) -> torch.Tensor:
    """Normalize image tensors into 3-channel [0, 1] batches for metric backends."""

    if images.ndim != 4:
        raise ValueError(f"Expected an NCHW image tensor, got shape {tuple(images.shape)}.")

    prepared = images.detach().float()
    if prepared.shape[1] == 1:
        prepared = prepared.repeat(1, 3, 1, 1)
    elif prepared.shape[1] > 3:
        prepared = prepared[:, :3]
    elif prepared.shape[1] != 3:
        raise ValueError(f"Unsupported image channel count for metrics: {prepared.shape[1]}")

    min_value = float(prepared.min())
    max_value = float(prepared.max())
    if min_value < -1.01 or max_value > 1.01:
        raise ValueError(
            "Metric images must be in either [-1, 1] or [0, 1]. "
            f"Observed range [{min_value}, {max_value}]."
        )
    if min_value < 0.0:
        prepared = ((prepared + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        prepared = prepared.clamp(0.0, 1.0)
    return prepared


def _load_torchvision_model(
    factory: Callable[..., nn.Module],
    *,
    weights: Any,
    allow_model_download: bool,
    factory_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Load a torchvision model from the local cache unless downloads are allowed."""

    kwargs = dict(factory_kwargs or {})
    if allow_model_download:
        return factory(weights=weights, **kwargs)

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_path = checkpoint_dir / Path(weights.url).name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Required model weights are not cached at {checkpoint_path}. "
            "Pre-download the weights on a login node or rerun with --allow-model-download."
        )

    model = factory(weights=None, **kwargs)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


class InceptionFeatureExtractor:
    """Extract pooled Inception features and logits for FID / IS computation."""

    feature_dim = 2048
    preprocess_signature = "inception_v3_imagenet_rgb299"

    def __init__(self, device: torch.device, *, allow_model_download: bool) -> None:
        self.device = device
        self.model = _load_torchvision_model(
            models.inception_v3,
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            allow_model_download=allow_model_download,
            factory_kwargs={"aux_logits": False, "transform_input": False},
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._captured_features: torch.Tensor | None = None
        self._hook = self.model.avgpool.register_forward_hook(self._capture_avgpool)

    def _capture_avgpool(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self._captured_features = output.flatten(1)

    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = prepare_images_for_metrics(images)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        batch = (batch.to(self.device) - mean) / std
        with torch.no_grad():
            logits = self.model(batch)
        if self._captured_features is None:
            raise RuntimeError("Inception feature hook did not capture any features.")
        return self._captured_features.detach().cpu(), logits.detach().cpu()


class AlexNetFeatureProxy:
    """Approximate LPIPS-style perceptual distances from AlexNet feature maps."""

    backend_name = "alexnet_feature_proxy"

    def __init__(self, device: torch.device, *, allow_model_download: bool) -> None:
        self.device = device
        self.model = _load_torchvision_model(
            models.alexnet,
            weights=AlexNet_Weights.IMAGENET1K_V1,
            allow_model_download=allow_model_download,
        ).features.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.layer_indices = {1, 4, 7, 9, 11}

    def compute(self, images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
        batch_a = prepare_images_for_metrics(images_a)
        batch_b = prepare_images_for_metrics(images_b)
        batch_a = F.interpolate(batch_a, size=(224, 224), mode="bilinear", align_corners=False)
        batch_b = F.interpolate(batch_b, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        batch_a = (batch_a.to(self.device) - mean) / std
        batch_b = (batch_b.to(self.device) - mean) / std

        total_distance = torch.zeros(batch_a.shape[0], device=self.device)
        feat_a = batch_a
        feat_b = batch_b
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model):
                feat_a = layer(feat_a)
                feat_b = layer(feat_b)
                if layer_idx in self.layer_indices:
                    feat_a = F.normalize(feat_a, dim=1, eps=1e-10)
                    feat_b = F.normalize(feat_b, dim=1, eps=1e-10)
                    total_distance = total_distance + (feat_a - feat_b).square().mean(dim=(1, 2, 3))
        return total_distance.detach().cpu()


class LPIPSDiversityMetric:
    """Compute LPIPS-style diversity across generated image pairs."""

    def __init__(self, device: torch.device, *, allow_model_download: bool) -> None:
        self.device = device
        self.backend_name = "alexnet_feature_proxy"
        self.description = (
            "Mean perceptual feature distance across generated-image pairs. "
            "Higher values indicate more diversity, not better fidelity."
        )
        self._lpips_model = None
        self._alexnet_proxy = None

        if importlib.util.find_spec("lpips") is not None:
            import lpips

            self._lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
            self._lpips_model.eval()
            self._lpips_model.requires_grad_(False)
            self.backend_name = "lpips_package_alex"
            self.description = (
                "Mean LPIPS distance across generated-image pairs. "
                "Higher values indicate more perceptual diversity, not better fidelity."
            )
        else:
            self._alexnet_proxy = AlexNetFeatureProxy(
                device,
                allow_model_download=allow_model_download,
            )

    def compute(self, images: torch.Tensor, pair_count: int | None = None) -> float:
        batch = prepare_images_for_metrics(images)
        max_pairs = batch.shape[0] // 2
        if max_pairs < 1:
            return float("nan")
        resolved_pairs = min(max_pairs, pair_count or max_pairs)
        batch = batch[: resolved_pairs * 2]
        images_a = batch[0::2]
        images_b = batch[1::2]

        if self._lpips_model is not None:
            with torch.no_grad():
                distances = self._lpips_model(
                    (images_a.to(self.device) * 2.0) - 1.0,
                    (images_b.to(self.device) * 2.0) - 1.0,
                )
            return float(distances.mean().item())

        if self._alexnet_proxy is None:  # pragma: no cover - defensive.
            raise RuntimeError("No LPIPS backend is available.")
        return float(self._alexnet_proxy.compute(images_a, images_b).mean().item())


@dataclass(frozen=True)
class FeatureStatistics:
    """Feature moments used for FID computation."""

    count: int
    mean: np.ndarray
    covariance: np.ndarray


class RunningFeatureStats:
    """Streaming accumulator for mean and covariance statistics."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.count = 0
        self.sum = np.zeros(feature_dim, dtype=np.float64)
        self.sum_outer = np.zeros((feature_dim, feature_dim), dtype=np.float64)

    def update(self, features: torch.Tensor) -> None:
        array = np.asarray(features.detach().cpu(), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features with shape (N, {self.feature_dim}), got {array.shape}."
            )
        self.count += array.shape[0]
        self.sum += array.sum(axis=0)
        self.sum_outer += array.T @ array

    def finalize(self) -> FeatureStatistics:
        if self.count < 2:
            raise ValueError("At least two samples are required to compute covariance statistics.")
        mean = self.sum / self.count
        covariance = (self.sum_outer - np.outer(self.sum, self.sum) / self.count) / (self.count - 1)
        return FeatureStatistics(
            count=self.count,
            mean=mean,
            covariance=covariance,
        )


def compute_fid(
    real_stats: FeatureStatistics,
    fake_stats: FeatureStatistics,
) -> float:
    """Compute the Frechet Inception Distance between two feature distributions."""

    mean_delta = real_stats.mean - fake_stats.mean
    cov_mean = linalg.sqrtm(real_stats.covariance @ fake_stats.covariance)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = (
        mean_delta @ mean_delta
        + np.trace(real_stats.covariance)
        + np.trace(fake_stats.covariance)
        - 2.0 * np.trace(cov_mean)
    )
    return float(max(fid, 0.0))


def compute_inception_score(
    logits: torch.Tensor,
    *,
    num_splits: int = 10,
) -> tuple[float, float]:
    """Compute the Inception Score mean and standard deviation."""

    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (N, C), got {tuple(logits.shape)}.")
    probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    split_probabilities = np.array_split(probabilities, min(num_splits, probabilities.shape[0]))
    split_scores: list[float] = []

    for split in split_probabilities:
        if split.size == 0:
            continue
        marginal = split.mean(axis=0, keepdims=True)
        kl = split * (np.log(split + 1e-12) - np.log(marginal + 1e-12))
        split_scores.append(float(np.exp(np.sum(kl, axis=1).mean())))

    if not split_scores:
        return float("nan"), float("nan")
    scores = np.asarray(split_scores, dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=0))


def save_feature_statistics(
    path: Path,
    stats: FeatureStatistics,
    metadata: dict[str, Any],
) -> None:
    """Persist reference statistics and their preprocessing metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        count=np.asarray(stats.count, dtype=np.int64),
        mean=stats.mean,
        covariance=stats.covariance,
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )


def load_feature_statistics(path: Path) -> tuple[FeatureStatistics, dict[str, Any]]:
    """Load cached reference statistics plus their metadata payload."""

    with np.load(path, allow_pickle=True) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        stats = FeatureStatistics(
            count=int(payload["count"].item()),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            covariance=np.asarray(payload["covariance"], dtype=np.float64),
        )
    return stats, metadata
