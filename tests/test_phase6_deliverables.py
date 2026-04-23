from __future__ import annotations

import json
from pathlib import Path

from diffusion.final_deliverables import (
    build_analysis_summary_markdown,
    build_artifact_index,
    build_best_run_rows,
    export_best_artifacts,
    export_final_deliverables,
)
from diffusion.parity_study import build_study_plans, execute_parity_suite, generate_final_study_summaries
from diffusion.recipes import load_recipe


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_option(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def fake_runner_with_artifacts(command: list[str], cwd: Path) -> None:
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
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "config_name": recipe.values["config_name"],
                    "protocol_name": recipe.values["protocol_name"],
                    "dataset_variant": recipe.values["dataset_variant"],
                    "image_size": recipe.values["image_size"],
                    "diffusion_channels": recipe.values["diffusion_channels"],
                    "diffusion_preprocessing": recipe.values["diffusion_preprocessing"],
                    "diffusion_backbone": recipe.values["diffusion_backbone"],
                    "prediction_type": recipe.values["prediction_type"],
                    "sampler": recipe.values["sampler"],
                    "sampling_steps": recipe.values["sampling_steps"],
                    "guidance_scale": recipe.values["guidance_scale"],
                }
            ),
            encoding="utf-8",
        )
        return

    if command[1] == "evaluate.py":
        checkpoint_path = Path(_parse_option(command, "--checkpoint"))
        run_root = checkpoint_path.parent.parent
        run_name = _parse_option(command, "--run-name")
        eval_dir = run_root / "evaluations" / run_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        dataset = run_root.parent.parent.name
        run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        seed_fragment = run_root.name.split("_seed")[-1]
        seed = int(seed_fragment) if seed_fragment.isdigit() else 0
        artifacts = {
            "generated_sample_grid": eval_dir / "generated_samples.png",
            "generated_samples": eval_dir / "generated_samples.png",
            "cfg_comparison_grid": eval_dir / "cfg.png",
            "diffusion_snapshots": eval_dir / "diffusion_snapshots.png",
            "reverse_process_snapshots": eval_dir / "diffusion_snapshots.png",
            "nearest_neighbor_grid": eval_dir / "nn.png",
            "reconstructions": eval_dir / "reconstructions.png",
            "reconstruction_preview": eval_dir / "reconstructions.png",
        }
        for artifact_path in artifacts.values():
            artifact_path.write_text(f"{dataset}:{artifact_path.name}", encoding="utf-8")
        metrics_payload = {
            "dataset": dataset,
            "config_name": run_manifest["config_name"],
            "protocol_name": run_manifest["protocol_name"],
            "dataset_variant": run_manifest["dataset_variant"],
            "seed": seed,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "evaluation_dir": str(eval_dir.resolve()),
            "metrics_path": str((eval_dir / "metrics.json").resolve()),
            "image_size": run_manifest["image_size"],
            "diffusion_channels": run_manifest["diffusion_channels"],
            "diffusion_preprocessing": run_manifest["diffusion_preprocessing"],
            "diffusion_backbone": run_manifest["diffusion_backbone"],
            "prediction_type": run_manifest["prediction_type"],
            "sampler": run_manifest["sampler"],
            "sampling_steps": run_manifest["sampling_steps"],
            "guidance_scale": run_manifest["guidance_scale"],
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
            "artifacts": {key: str(path.resolve()) for key, path in artifacts.items()},
        }
        (eval_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        (eval_dir / "evaluation_manifest.json").write_text(json.dumps({"evaluation": run_name}), encoding="utf-8")
        return

    raise AssertionError(f"Unexpected command: {command}")


def _create_completed_study(tmp_path: Path) -> Path:
    study_dir = tmp_path / "study"
    plans = build_study_plans(
        study_dir=study_dir,
        data_dir=tmp_path / "data",
        datasets=("mnist", "fashion", "cifar10"),
        seeds=(1, 2),
    )
    execute_parity_suite(
        study_dir=study_dir,
        plans=plans,
        phase="both",
        skip_existing=False,
        runner=fake_runner_with_artifacts,
        repo_root=REPO_ROOT,
    )
    return study_dir


def test_final_deliverables_bundle_schema(tmp_path: Path) -> None:
    study_dir = _create_completed_study(tmp_path)
    payload = export_final_deliverables(study_dir=study_dir)

    assert Path(payload["bundle_manifest"]["json"]).exists()
    assert Path(payload["main_results_table"]["csv"]).exists()
    assert Path(payload["mean_std_table"]["csv"]).exists()
    assert Path(payload["best_runs_table"]["csv"]).exists()
    assert Path(payload["artifact_index"]["csv"]).exists()
    assert Path(payload["analysis_summary_path"]).exists()
    assert Path(payload["project_report_summary_path"]).exists()
    assert Path(payload["presentation_figure_index_path"]).exists()


def test_artifact_indexing_and_export_integrity(tmp_path: Path) -> None:
    study_dir = _create_completed_study(tmp_path)
    summary_payload = generate_final_study_summaries(study_dir=study_dir)
    best_run_rows = build_best_run_rows(summary_payload["selection_payload"])
    artifact_index_rows = build_artifact_index(best_run_rows)
    exported_rows = export_best_artifacts(
        artifact_index_rows,
        output_dir=tmp_path / "exported_figures",
    )

    assert any(row["artifact_key"] == "generated_sample_grid" for row in exported_rows)
    for row in exported_rows:
        if row["source_path"] is not None:
            assert Path(str(row["source_path"])).exists()
            assert row["exported_path"] is not None
            assert Path(str(row["exported_path"])).exists()


def test_markdown_summary_generation_contains_required_sections(tmp_path: Path) -> None:
    study_dir = _create_completed_study(tmp_path)
    summary_payload = generate_final_study_summaries(study_dir=study_dir)
    best_run_rows = build_best_run_rows(summary_payload["selection_payload"])
    markdown = build_analysis_summary_markdown(
        study_dir=study_dir,
        main_results_rows=best_run_rows,
        mean_std_rows=summary_payload["per_dataset_rows"],
        best_run_rows=best_run_rows,
    )

    assert "## Experimental Setup" in markdown
    assert "## Study Defaults" in markdown
    assert "## Main Findings By Dataset" in markdown
    assert "## CFG Observations" in markdown
    assert "## Limitations" in markdown
    assert "## Future Work" in markdown


def test_best_run_export_integrity(tmp_path: Path) -> None:
    study_dir = _create_completed_study(tmp_path)
    payload = export_final_deliverables(study_dir=study_dir, output_dir=tmp_path / "deliverables")

    exported = payload["exported_artifacts"]
    mnist_samples = next(
        row for row in exported
        if row["dataset"] == "mnist" and row["artifact_key"] == "generated_sample_grid"
    )
    source_text = Path(str(mnist_samples["source_path"])).read_text(encoding="utf-8")
    exported_text = Path(str(mnist_samples["exported_path"])).read_text(encoding="utf-8")
    assert source_text == exported_text
