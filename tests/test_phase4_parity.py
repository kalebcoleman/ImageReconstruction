from __future__ import annotations

import json
from pathlib import Path

from aggregate_results import aggregate_evaluation_results, save_aggregate_outputs
from diffusion.data import describe_diffusion_preprocessing
from diffusion.recipes import load_recipe
from diffusion.reporting import save_manifest_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_recipe_loading_and_merging() -> None:
    recipe = load_recipe(REPO_ROOT / "configs" / "diffusion" / "mnist_64.yaml")

    assert recipe.values["config_name"] == "mnist_64"
    assert recipe.values["dataset"] == "mnist"
    assert recipe.values["image_size"] == 64
    assert recipe.values["diffusion_channels"] == 3
    assert recipe.values["prediction_type"] == "v"
    assert recipe.values["diffusion_preprocessing"] == "parity_64"
    assert recipe.values["protocol_name"] == "adm64_parity_v1"


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


def test_dataset_parity_configs_resolve_same_locked_geometry() -> None:
    recipe_names = ("mnist_64", "fashion_64", "cifar10_64", "imagenet_64")
    loaded = [
        load_recipe(REPO_ROOT / "configs" / "diffusion" / f"{recipe_name}.yaml")
        for recipe_name in recipe_names
    ]

    assert {recipe.values["image_size"] for recipe in loaded} == {64}
    assert {recipe.values["diffusion_channels"] for recipe in loaded} == {3}
    assert {recipe.values["diffusion_backbone"] for recipe in loaded} == {"adm"}
    assert {recipe.values["prediction_type"] for recipe in loaded} == {"v"}
    assert {recipe.values["sampler"] for recipe in loaded} == {"ddim"}
    assert {recipe.values["diffusion_preprocessing"] for recipe in loaded} == {"parity_64"}


def test_shared_preprocessing_assumptions_are_explicit() -> None:
    for dataset_name in ("mnist", "fashion", "cifar10", "imagenet"):
        description = describe_diffusion_preprocessing(
            dataset_name,
            image_size=64,
            channels=3,
            preprocessing_protocol="parity_64",
        )
        assert description["protocol"] == "parity_64"
        assert description["image_size"] == 64
        assert description["channels"] == 3
        assert description["deterministic_train_preprocessing"] is True
        assert all("random" not in str(operation) for operation in description["train_ops"])


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
        "config_name": "mnist_64",
        "protocol_name": "adm64_parity_v1",
        "dataset_variant": "mnist_64_rgb_parity",
        "checkpoint_path": "/tmp/checkpoints/best.pt",
        "evaluation_dir": str(evaluation_root),
        "image_size": 64,
        "diffusion_channels": 3,
        "diffusion_preprocessing": "parity_64",
        "diffusion_backbone": "adm",
        "prediction_type": "v",
        "sampler": "ddim",
        "sampling_steps": 50,
        "guidance_scale": 3.0,
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
            "generated_sample_grid": "/tmp/generated.png",
            "cfg_comparison_grid": "/tmp/cfg.png",
            "reverse_process_snapshots": "/tmp/snapshots.png",
            "nearest_neighbor_grid": "/tmp/nn.png",
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    rows = aggregate_evaluation_results([metrics_path])
    output_paths = save_aggregate_outputs(tmp_path / "aggregate", rows)

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "mnist"
    assert row["protocol_name"] == "adm64_parity_v1"
    assert row["fid"] == 10.0
    assert Path(output_paths["json"]).exists()
    assert Path(output_paths["csv"]).exists()
    assert Path(output_paths["markdown"]).exists()
    assert Path(output_paths["markdown_table"]).exists()
