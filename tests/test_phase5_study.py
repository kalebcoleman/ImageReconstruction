from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_parity_suite import _parse_seeds
from diffusion.parity_study import (
    DEFAULT_STUDY_SEEDS,
    DEFAULT_SMOKE_STUDY_SEEDS,
    FINAL_STUDY_DATASETS,
    build_study_plans,
    ensure_path_safe,
    execute_parity_suite,
    generate_final_study_summaries,
    load_registry,
    select_best_and_median_runs,
    write_study_plan_files,
)
from diffusion.recipes import load_recipe


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_option(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def fake_runner(command: list[str], cwd: Path) -> None:
    del cwd
    if command[1] == "train.py":
        recipe_path = Path(_parse_option(command, "--config"))
        recipe = load_recipe(recipe_path)
        dataset = recipe.values["dataset"]
        output_dir = Path(_parse_option(command, "--output-dir"))
        run_name = _parse_option(command, "--run-name")
        run_dir = output_dir / dataset / "diffusion" / run_name
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps({"dataset": dataset}), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(json.dumps({"run_name": run_name}), encoding="utf-8")
        return

    if command[1] == "evaluate.py":
        checkpoint_path = Path(_parse_option(command, "--checkpoint"))
        run_root = checkpoint_path.parent.parent
        run_name = _parse_option(command, "--run-name")
        eval_dir = run_root / "evaluations" / run_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        dataset = run_root.parent.parent.name
        seed_fragment = run_root.name.split("_seed")[-1]
        seed = int(seed_fragment) if seed_fragment.isdigit() else 0
        metrics_payload = {
            "dataset": dataset,
            "config_name": run_root.name.split("_seed")[0].removeprefix("parity_"),
            "protocol_name": "adm64_parity_v1",
            "dataset_variant": f"{dataset}_64_rgb_parity",
            "seed": seed,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "evaluation_dir": str(eval_dir.resolve()),
            "metrics_path": str((eval_dir / "metrics.json").resolve()),
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
                "fid": float(seed + 10),
                "inception_score_mean": 1.0 + seed,
                "inception_score_std": 0.1,
                "lpips_diversity": 0.2 + seed * 0.01,
            },
            "paired_metrics": {
                "psnr": 20.0,
                "ssim": 0.8,
            },
            "artifacts": {
                "generated_sample_grid": str((eval_dir / "generated.png").resolve()),
                "cfg_comparison_grid": str((eval_dir / "cfg.png").resolve()),
                "reverse_process_snapshots": str((eval_dir / "snapshots.png").resolve()),
                "nearest_neighbor_grid": str((eval_dir / "nn.png").resolve()),
            },
        }
        (eval_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        (eval_dir / "evaluation_manifest.json").write_text(json.dumps({"evaluation": run_name}), encoding="utf-8")
        return

    raise AssertionError(f"Unexpected command: {command}")


def test_build_study_plans_defaults_and_multi_seed_resolution(tmp_path: Path) -> None:
    plans = build_study_plans(
        study_dir=tmp_path / "study",
        data_dir=tmp_path / "data",
    )

    assert len(plans) == len(FINAL_STUDY_DATASETS) * len(DEFAULT_STUDY_SEEDS)
    assert {plan.dataset for plan in plans} == set(FINAL_STUDY_DATASETS)
    assert {plan.seed for plan in plans} == set(DEFAULT_STUDY_SEEDS)
    assert all(plan.run_name.startswith("parity_") for plan in plans)


def test_smoke_seed_defaults_are_single_seed() -> None:
    assert _parse_seeds(None, smoke=True) == DEFAULT_SMOKE_STUDY_SEEDS


def test_build_smoke_study_plans_resolve_lightweight_recipes_and_eval_overrides(tmp_path: Path) -> None:
    plans = build_study_plans(
        study_dir=tmp_path / "study",
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=DEFAULT_SMOKE_STUDY_SEEDS,
        smoke=True,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.study_mode == "smoke"
    assert plan.config_name == "mnist_64_smoke"
    assert str(plan.recipe_path).endswith("configs/diffusion/smoke/mnist_64.yaml")
    assert plan.run_name == "parity_mnist_64_smoke_seed001"
    assert _parse_option(plan.eval_command, "--num-generated-samples") == "64"
    assert _parse_option(plan.eval_command, "--artifact-sample-count") == "4"
    assert _parse_option(plan.eval_command, "--nearest-neighbor-count") == "4"
    assert _parse_option(plan.eval_command, "--nearest-neighbor-reference-limit") == "256"
    assert _parse_option(plan.eval_command, "--lpips-pair-count") == "16"


def test_build_study_plans_can_pass_allow_model_download_to_eval(tmp_path: Path) -> None:
    plans = build_study_plans(
        study_dir=tmp_path / "study",
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1,),
        allow_model_download=True,
    )

    assert len(plans) == 1
    assert "--allow-model-download" in plans[0].eval_command


def test_write_study_plan_files_outputs_shell_and_slurm(tmp_path: Path) -> None:
    plans = build_study_plans(
        study_dir=tmp_path / "study",
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1, 2),
    )
    outputs = write_study_plan_files(
        study_dir=tmp_path / "study",
        plans=plans,
        phase="both",
    )

    assert Path(outputs["commands_txt"]).exists()
    assert Path(outputs["shell_script"]).exists()
    assert Path(outputs["slurm_script"]).exists()


def test_execute_parity_suite_links_train_and_eval_outputs(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    plans = build_study_plans(
        study_dir=study_dir,
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1,),
    )

    registry = execute_parity_suite(
        study_dir=study_dir,
        plans=plans,
        phase="both",
        skip_existing=False,
        runner=fake_runner,
        repo_root=REPO_ROOT,
    )

    entry = next(iter(registry["entries"].values()))
    assert entry["train_status"] == "completed"
    assert entry["eval_status"] == "completed"
    assert entry["selected_checkpoint_path"] is not None
    assert entry["selected_evaluation_dir"] is not None
    assert Path(entry["train_manifest_path"]).exists()
    assert Path(entry["evaluation_manifest_path"]).exists()


def test_skip_existing_behavior_avoids_rerun(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    plans = build_study_plans(
        study_dir=study_dir,
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1,),
    )
    plan = plans[0]
    fake_runner(plan.train_command, REPO_ROOT)
    fake_runner(plan.eval_command, REPO_ROOT)

    def unexpected_runner(command: list[str], cwd: Path) -> None:  # pragma: no cover - defensive
        raise AssertionError(f"Runner should not have been called: {command} @ {cwd}")

    registry = execute_parity_suite(
        study_dir=study_dir,
        plans=plans,
        phase="both",
        skip_existing=True,
        runner=unexpected_runner,
        repo_root=REPO_ROOT,
    )

    entry = next(iter(registry["entries"].values()))
    assert entry["train_status"] == "skipped_existing"
    assert entry["eval_status"] == "skipped_existing"
    assert entry["selected_checkpoint_path"] is not None
    assert entry["selected_evaluation_dir"] is not None


def test_output_path_safety_rejects_incomplete_existing_run(tmp_path: Path) -> None:
    plans = build_study_plans(
        study_dir=tmp_path / "study",
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1,),
    )
    plan = plans[0]
    plan.train_run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileExistsError):
        ensure_path_safe(plan, phase="train", skip_existing=False)


def test_smoke_and_full_studies_use_distinct_run_paths_and_cannot_share_registry(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    full_plan = build_study_plans(
        study_dir=study_dir,
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1,),
    )[0]
    smoke_plan = build_study_plans(
        study_dir=study_dir,
        data_dir=tmp_path / "data",
        datasets=("mnist",),
        seeds=(1,),
        smoke=True,
    )[0]

    assert full_plan.run_name == "parity_mnist_64_seed001"
    assert smoke_plan.run_name == "parity_mnist_64_smoke_seed001"
    assert full_plan.train_run_dir != smoke_plan.train_run_dir
    assert full_plan.evaluation_dir != smoke_plan.evaluation_dir

    execute_parity_suite(
        study_dir=study_dir,
        plans=[full_plan],
        phase="train",
        skip_existing=False,
        runner=fake_runner,
        repo_root=REPO_ROOT,
    )

    with pytest.raises(ValueError, match="separate study directories"):
        execute_parity_suite(
            study_dir=study_dir,
            plans=[smoke_plan],
            phase="train",
            skip_existing=False,
            runner=fake_runner,
            repo_root=REPO_ROOT,
        )


def test_best_run_selection_logic() -> None:
    rows = [
        {"dataset": "mnist", "seed": 1, "fid": 12.0, "metrics_path": "/tmp/a"},
        {"dataset": "mnist", "seed": 2, "fid": 10.0, "metrics_path": "/tmp/b"},
        {"dataset": "mnist", "seed": 3, "fid": 11.0, "metrics_path": "/tmp/c"},
    ]
    selections = select_best_and_median_runs(rows)

    assert selections["mnist"]["best"]["fid"] == 10.0
    assert selections["mnist"]["median"]["fid"] == 11.0


def test_final_study_summary_schema(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    plans = build_study_plans(
        study_dir=study_dir,
        data_dir=tmp_path / "data",
        datasets=("mnist", "fashion"),
        seeds=(1, 2),
    )
    execute_parity_suite(
        study_dir=study_dir,
        plans=plans,
        phase="both",
        skip_existing=False,
        runner=fake_runner,
        repo_root=REPO_ROOT,
    )

    payload = generate_final_study_summaries(study_dir=study_dir)

    assert Path(payload["report_path"]).exists()
    assert len(payload["per_run_rows"]) == 4
    assert len(payload["per_dataset_rows"]) == 2
    mnist_summary = next(row for row in payload["per_dataset_rows"] if row["dataset"] == "mnist")
    assert "best_generated_sample_grid" in mnist_summary
    assert mnist_summary["best_fid"] is not None
