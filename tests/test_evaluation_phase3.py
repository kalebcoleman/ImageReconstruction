from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion.eval_metrics import _load_torchvision_model, prepare_images_for_metrics
from diffusion.eval_pipeline import (
    CheckpointEvaluationConfig,
    _load_or_compute_reference_stats,
    _save_nearest_neighbor_grid,
    run_checkpoint_evaluation,
)
from train import ExperimentConfig, instantiate_model, json_ready


class DummyInception:
    feature_dim = 4
    preprocess_signature = "dummy_inception_backend"

    def __init__(self) -> None:
        self.extract_calls = 0

    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.extract_calls += 1
        prepared = prepare_images_for_metrics(images)
        channel_means = prepared.mean(dim=(2, 3))
        features = torch.cat([channel_means, channel_means.mean(dim=1, keepdim=True)], dim=1)
        logits = torch.cat([features, torch.zeros(prepared.shape[0], 6)], dim=1)
        return features, logits


class DummyLPIPS:
    backend_name = "dummy_lpips"
    description = "Dummy LPIPS proxy for tests."

    def compute(self, images: torch.Tensor, pair_count: int | None = None) -> float:
        del pair_count
        prepared = prepare_images_for_metrics(images)
        return float(prepared.mean().item())


class DummyMetricBackend:
    def __init__(self) -> None:
        self.inception = DummyInception()
        self.lpips = DummyLPIPS()


class DummyVisionModel(torch.nn.Module):
    def __init__(self, *, aux_logits: bool) -> None:
        super().__init__()
        self.aux_logits = aux_logits
        self.AuxLogits = object() if aux_logits else None
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.loaded_state_dict = state_dict


def dummy_metric_backend_factory(
    device: torch.device,
    *,
    allow_model_download: bool,
) -> DummyMetricBackend:
    del device, allow_model_download
    return DummyMetricBackend()


def make_dummy_loader() -> DataLoader:
    images = torch.rand(6, 3, 64, 64) * 2.0 - 1.0
    labels = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long) % 10
    return DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)


def make_dummy_checkpoint(tmp_path: Path) -> Path:
    config = ExperimentConfig(
        model="diffusion",
        dataset="mnist",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        epochs=1,
        batch_size=2,
        num_workers=0,
        download=False,
        diffusion_backbone="adm",
        image_size=64,
        diffusion_channels=3,
        dataset_num_classes=10,
        timesteps=4,
        base_channels=8,
        time_dim=32,
        num_res_blocks=1,
        prediction_type="eps",
        sampler="ddim",
        sampling_steps=4,
        guidance_scale=1.0,
        amp_dtype="none",
        class_dropout_prob=0.1,
    )
    model, _ = instantiate_model(config, torch.device("cpu"))
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None,
            "evaluation_weights": "model",
            "config": json_ready(asdict(config)),
            "metrics": {},
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_prepare_images_for_metrics_handles_range_and_channels() -> None:
    grayscale = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32).view(1, 1, 4, 4)
    prepared = prepare_images_for_metrics(grayscale)

    assert prepared.shape == (1, 3, 4, 4)
    assert float(prepared.min()) >= 0.0
    assert float(prepared.max()) <= 1.0


def test_load_torchvision_model_disables_aux_logits_after_weighted_load() -> None:
    calls: list[dict[str, object]] = []

    def factory(*, weights: object | None = None, aux_logits: bool = True, transform_input: bool = False) -> DummyVisionModel:
        calls.append(
            {
                "weights": weights,
                "aux_logits": aux_logits,
                "transform_input": transform_input,
            }
        )
        return DummyVisionModel(aux_logits=aux_logits)

    model = _load_torchvision_model(
        factory,
        weights=object(),
        allow_model_download=True,
        factory_kwargs={"aux_logits": False, "transform_input": False},
    )

    assert len(calls) == 1
    assert calls[0]["weights"] is not None
    assert calls[0]["aux_logits"] is True
    assert model.aux_logits is False
    assert model.AuxLogits is None


def test_reference_stats_cache_reuse(monkeypatch, tmp_path: Path) -> None:
    backend = DummyMetricBackend()
    config = ExperimentConfig(
        model="diffusion",
        dataset="mnist",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        diffusion_backbone="adm",
        image_size=64,
        diffusion_channels=3,
        dataset_num_classes=10,
        timesteps=4,
        base_channels=8,
        time_dim=32,
        num_res_blocks=1,
        amp_dtype="none",
    )
    monkeypatch.setattr("diffusion.eval_pipeline._build_loader", lambda *args, **kwargs: make_dummy_loader())

    stats_dir = tmp_path / "reference_stats"
    stats, stats_path = _load_or_compute_reference_stats(
        config,
        backend=backend,
        reference_stats_dir=stats_dir,
        batch_size=2,
    )
    first_call_count = backend.inception.extract_calls
    assert stats.count == 6
    assert stats_path.exists()

    backend.inception.extract_calls = 0
    cached_stats, cached_path = _load_or_compute_reference_stats(
        config,
        backend=backend,
        reference_stats_dir=stats_dir,
        batch_size=2,
    )
    assert cached_path == stats_path
    assert backend.inception.extract_calls == 0
    assert cached_stats.count == stats.count


def test_checkpoint_only_evaluation_path(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = make_dummy_checkpoint(tmp_path)
    monkeypatch.setattr("diffusion.eval_pipeline._build_loader", lambda *args, **kwargs: make_dummy_loader())

    payload = run_checkpoint_evaluation(
        CheckpointEvaluationConfig(
            checkpoint_path=checkpoint_path,
            mode="evaluate",
            output_dir=tmp_path / "evaluations",
            reference_stats_dir=tmp_path / "reference_stats",
            num_generated_samples=4,
            eval_batch_size=2,
            artifact_sample_count=2,
            num_workers=0,
            amp_dtype="none",
        ),
        device=torch.device("cpu"),
        metric_backend_factory=dummy_metric_backend_factory,
    )

    assert Path(payload["metrics_path"]).exists()
    assert Path(payload["summary_path"]).exists()
    assert payload["generative_metrics"] is not None
    assert "fid" in payload["generative_metrics"]
    assert "inception_score_mean" in payload["generative_metrics"]
    assert "lpips_diversity" in payload["generative_metrics"]
    assert payload["paired_metrics"] is not None
    assert "generated_sample_grid" in payload["artifacts"]
    assert "nearest_neighbor_grid" in payload["artifacts"]
    assert Path(payload["artifacts"]["nearest_neighbor_grid"]).exists()


def test_ddpm_and_ddim_checkpoint_sampling_paths(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = make_dummy_checkpoint(tmp_path)

    payload_ddpm = run_checkpoint_evaluation(
        CheckpointEvaluationConfig(
            checkpoint_path=checkpoint_path,
            mode="sample",
            output_dir=tmp_path / "sample_ddpm",
            sampler="ddpm",
            amp_dtype="none",
            num_generated_samples=2,
            artifact_sample_count=2,
        ),
        device=torch.device("cpu"),
        metric_backend_factory=dummy_metric_backend_factory,
    )
    payload_ddim = run_checkpoint_evaluation(
        CheckpointEvaluationConfig(
            checkpoint_path=checkpoint_path,
            mode="sample",
            output_dir=tmp_path / "sample_ddim",
            sampler="ddim",
            sampling_steps=4,
            amp_dtype="none",
            num_generated_samples=2,
            artifact_sample_count=2,
        ),
        device=torch.device("cpu"),
        metric_backend_factory=dummy_metric_backend_factory,
    )

    assert payload_ddpm["sampler"] == "ddpm"
    assert payload_ddim["sampler"] == "ddim"
    assert Path(payload_ddpm["artifacts"]["generated_sample_grid"]).exists()
    assert Path(payload_ddim["artifacts"]["generated_sample_grid"]).exists()


def test_nearest_neighbor_artifact_generation(tmp_path: Path, monkeypatch) -> None:
    config = ExperimentConfig(
        model="diffusion",
        dataset="mnist",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        diffusion_backbone="adm",
        image_size=64,
        diffusion_channels=3,
        dataset_num_classes=10,
        timesteps=4,
        base_channels=8,
        time_dim=32,
        num_res_blocks=1,
    )
    monkeypatch.setattr("diffusion.eval_pipeline._build_loader", lambda *args, **kwargs: make_dummy_loader())
    generated_images = torch.rand(2, 3, 64, 64)
    save_path = tmp_path / "nearest_neighbors.png"

    _save_nearest_neighbor_grid(
        generated_images,
        config=config,
        save_path=save_path,
        batch_size=2,
        max_reference_images=4,
    )

    assert save_path.exists()
