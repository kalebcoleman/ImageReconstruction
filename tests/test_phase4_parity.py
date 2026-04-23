from __future__ import annotations

import json
from pathlib import Path

from aggregate_results import aggregate_evaluation_results, save_aggregate_outputs
from diffusion.data import describe_diffusion_preprocessing
from diffusion.recipes import load_recipe
from diffusion.reporting import save_manifest_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_recipe_loading_and_merging() -> None:
    recipe = load_recipe(REPO_ROOT / "configs" / "diffusion" / "mnist.yaml")

    assert recipe.values["config_name"] == "mnist"
    assert recipe.values["dataset"] == "mnist"
    assert recipe.values["image_size"] == 28
    assert recipe.values["diffusion_channels"] == 1
    assert recipe.values["diffusion_backbone"] == "legacy"
    assert recipe.values["prediction_type"] == "eps"
    assert recipe.values["diffusion_preprocessing"] == "default"
    assert recipe.values["protocol_name"] == "legacy28_gray_v1"


def test_recipe_loader_rejects_locked_override(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "base.yaml").write_text(
        "\n".join(
            [
                "config_name: base",
                "image_size: 64",
                "diffusion_channels: 3",
                "protocol:",
                "  name: locked_protocol",
                "  allowed_overrides:",
                "    - dataset",
                "    - batch_size",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (configs_dir / "child.yaml").write_text(
        "\n".join(
            [
                "inherits:",
                "  - base.yaml",
                "dataset: mnist",
                "image_size: 32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_recipe(configs_dir / "child.yaml")
    except ValueError as exc:
        assert "locked settings" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected the child recipe to be rejected.")


def test_final_study_configs_resolve_dataset_appropriate_defaults() -> None:
    recipes = {
        recipe_name: load_recipe(REPO_ROOT / "configs" / "diffusion" / f"{recipe_name}.yaml")
        for recipe_name in ("mnist", "fashion", "cifar10")
    }

    assert recipes["mnist"].values["diffusion_backbone"] == "legacy"
    assert recipes["mnist"].values["image_size"] == 28
    assert recipes["mnist"].values["diffusion_channels"] == 1
    assert recipes["mnist"].values["diffusion_preprocessing"] == "default"

    assert recipes["fashion"].values["diffusion_backbone"] == "legacy"
    assert recipes["fashion"].values["image_size"] == 28
    assert recipes["fashion"].values["diffusion_channels"] == 1
    assert recipes["fashion"].values["diffusion_preprocessing"] == "default"

    assert recipes["cifar10"].values["diffusion_backbone"] == "adm"
    assert recipes["cifar10"].values["image_size"] == 64
    assert recipes["cifar10"].values["diffusion_channels"] == 3
    assert recipes["cifar10"].values["diffusion_preprocessing"] == "default"


def test_smoke_configs_resolve_lightweight_dataset_appropriate_defaults() -> None:
    recipes = {
        recipe_name: load_recipe(REPO_ROOT / "configs" / "diffusion" / "smoke" / f"{recipe_name}.yaml")
        for recipe_name in ("mnist", "fashion", "cifar10")
    }

    assert recipes["mnist"].values["protocol_name"] == "legacy28_gray_smoke_v1"
    assert recipes["mnist"].values["image_size"] == 28
    assert recipes["mnist"].values["diffusion_channels"] == 1
    assert recipes["mnist"].values["diffusion_backbone"] == "legacy"
    assert recipes["mnist"].values["timesteps"] == 100
    assert recipes["mnist"].values["sampling_steps"] == 20
    assert recipes["mnist"].values["epochs"] == 1
    assert recipes["mnist"].values["sample_count"] == 4
    assert recipes["mnist"].values["eval_num_generated_samples"] == 64

    assert recipes["fashion"].values["protocol_name"] == "legacy28_gray_smoke_v1"
    assert recipes["fashion"].values["image_size"] == 28
    assert recipes["fashion"].values["diffusion_channels"] == 1
    assert recipes["fashion"].values["diffusion_backbone"] == "legacy"

    assert recipes["cifar10"].values["protocol_name"] == "adm64_rgb_smoke_v1"
    assert recipes["cifar10"].values["image_size"] == 64
    assert recipes["cifar10"].values["diffusion_channels"] == 3
    assert recipes["cifar10"].values["diffusion_backbone"] == "adm"
    assert recipes["cifar10"].values["diffusion_preprocessing"] == "default"
    assert all(str(recipe.values["config_name"]).endswith("_smoke") for recipe in recipes.values())


def test_dataset_appropriate_preprocessing_assumptions_are_explicit() -> None:
    mnist_description = describe_diffusion_preprocessing(
        "mnist",
        image_size=28,
        channels=1,
        preprocessing_protocol="default",
    )
    assert mnist_description["protocol"] == "default"
    assert mnist_description["image_size"] == 28
    assert mnist_description["channels"] == 1
    assert mnist_description["channel_conversion"] == "1->1"
    assert mnist_description["train_ops"] == ["resize(28x28)"]

    fashion_description = describe_diffusion_preprocessing(
        "fashion",
        image_size=28,
        channels=1,
        preprocessing_protocol="default",
    )
    assert fashion_description["channel_conversion"] == "1->1"

    cifar_description = describe_diffusion_preprocessing(
        "cifar10",
        image_size=64,
        channels=3,
        preprocessing_protocol="default",
    )
    assert cifar_description["protocol"] == "default"
    assert cifar_description["image_size"] == 64
    assert cifar_description["channels"] == 3
    assert cifar_description["channel_conversion"] == "3->3"
    assert "random_horizontal_flip" in cifar_description["train_ops"]


def test_manifest_generation_outputs_all_formats(tmp_path: Path) -> None:
    paths = save_manifest_bundle(
        tmp_path,
        basename="run_manifest",
        title="Run Manifest",
        payload={"dataset": "mnist", "fid": 12.34},
    )

    assert Path(paths["json"]).exists()
    assert Path(paths["yaml"]).exists()
    assert Path(paths["markdown"]).exists()


def test_aggregate_results_output_schema(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "outputs" / "mnist" / "diffusion" / "run_a" / "evaluations" / "eval_a"
    evaluation_root.mkdir(parents=True)
    metrics_path = evaluation_root / "metrics.json"
    metrics_payload = {
        "dataset": "mnist",
        "config_name": "mnist",
        "protocol_name": "legacy28_gray_v1",
        "dataset_variant": "mnist_28_gray_legacy",
        "checkpoint_path": "/tmp/checkpoints/best.pt",
        "evaluation_dir": str(evaluation_root),
        "image_size": 28,
        "diffusion_channels": 1,
        "diffusion_preprocessing": "default",
        "diffusion_backbone": "legacy",
        "prediction_type": "eps",
        "sampler": "ddim",
        "sampling_steps": 50,
        "guidance_scale": 1.0,
        "model_parameters": 123456,
        "generative_metrics": {
            "fid": 10.0,
            "inception_score_mean": 1.5,
            "inception_score_std": 0.1,
            "lpips_diversity": 0.25,
        },
        "paired_metrics": {
            "psnr": 20.0,
            "ssim": 0.8,
        },
        "artifacts": {
            "generated_sample_grid": "/tmp/generated_samples.png",
            "generated_samples": "/tmp/generated_samples.png",
            "cfg_comparison_grid": "/tmp/cfg.png",
            "diffusion_snapshots": "/tmp/diffusion_snapshots.png",
            "reverse_process_snapshots": "/tmp/diffusion_snapshots.png",
            "nearest_neighbor_grid": "/tmp/nn.png",
            "reconstructions": "/tmp/reconstructions.png",
            "reconstruction_preview": "/tmp/reconstructions.png",
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    rows = aggregate_evaluation_results([metrics_path])
    output_paths = save_aggregate_outputs(tmp_path / "aggregate", rows)

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "mnist"
    assert row["protocol_name"] == "legacy28_gray_v1"
    assert row["fid"] == 10.0
    assert row["diffusion_snapshots"] == "/tmp/diffusion_snapshots.png"
    assert row["reconstructions"] == "/tmp/reconstructions.png"
    assert Path(output_paths["json"]).exists()
    assert Path(output_paths["csv"]).exists()
    assert Path(output_paths["markdown"]).exists()
    assert Path(output_paths["markdown_table"]).exists()
