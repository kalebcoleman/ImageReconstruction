from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Callable

from aggregate_results import aggregate_evaluation_results, discover_metric_files, save_aggregate_outputs
from diffusion.data import normalize_dataset_name
from diffusion.recipes import load_recipe
from diffusion.reporting import save_manifest_bundle


FINAL_STUDY_DATASETS: tuple[str, ...] = ("mnist", "fashion", "cifar10")
DEFAULT_STUDY_SEEDS: tuple[int, ...] = (1, 2, 3)
DEFAULT_SMOKE_STUDY_SEEDS: tuple[int, ...] = (1,)
DEFAULT_STUDY_CONFIG_DIR = Path("configs/diffusion")
SMOKE_STUDY_CONFIG_DIR = DEFAULT_STUDY_CONFIG_DIR / "smoke"
SMOKE_EVAL_COMMAND_OVERRIDES: tuple[str, ...] = (
    "--num-generated-samples",
    "64",
    "--artifact-sample-count",
    "4",
    "--nearest-neighbor-count",
    "4",
    "--nearest-neighbor-reference-limit",
    "256",
    "--lpips-pair-count",
    "16",
)
SUPPORTED_PHASES: tuple[str, ...] = ("train", "eval", "both")
SUPPORTED_STUDY_MODES: tuple[str, ...] = ("full", "smoke")


@dataclass(frozen=True)
class StudyRunPlan:
    """Deterministic train/eval plan for one dataset and seed."""

    study_mode: str
    dataset: str
    seed: int
    recipe_path: Path
    config_name: str
    protocol_name: str | None
    study_dir: Path
    data_dir: Path
    runs_root: Path
    summaries_root: Path
    run_name: str
    eval_run_name: str
    train_run_dir: Path
    checkpoint_path: Path
    evaluation_dir: Path
    evaluation_metrics_path: Path
    train_command: list[str]
    eval_command: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def build_study_run_name(config_name: str, seed: int) -> str:
    """Return the deterministic run name used by the final-study suite."""

    return f"study_{config_name}_seed{seed:03d}"


def build_recipe_path(config_dir: Path, dataset: str) -> Path:
    normalized_dataset = normalize_dataset_name(dataset)
    candidate_names = (
        f"{normalized_dataset}.yaml",
        f"{normalized_dataset}_64.yaml",
    )
    for candidate_name in candidate_names:
        recipe_path = config_dir / candidate_name
        if recipe_path.exists():
            return recipe_path
    raise FileNotFoundError(
        f"Could not find a study recipe for {dataset} in {config_dir}. "
        f"Tried {list(candidate_names)}."
    )


def resolve_study_config_dir(config_dir: Path, *, smoke: bool) -> Path:
    """Map the default study recipe directory to the smoke recipe directory when requested."""

    if smoke and config_dir == DEFAULT_STUDY_CONFIG_DIR:
        return SMOKE_STUDY_CONFIG_DIR
    return config_dir


def resolve_study_mode(*, smoke: bool) -> str:
    return "smoke" if smoke else "full"


def build_study_plans(
    *,
    study_dir: Path,
    data_dir: Path,
    datasets: tuple[str, ...] = FINAL_STUDY_DATASETS,
    seeds: tuple[int, ...] = DEFAULT_STUDY_SEEDS,
    config_dir: Path = DEFAULT_STUDY_CONFIG_DIR,
    smoke: bool = False,
    allow_model_download: bool = False,
    repo_root: Path | None = None,
) -> list[StudyRunPlan]:
    """Create deterministic train/eval plans for the final study."""

    resolved_repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    resolved_study_dir = study_dir.expanduser().resolve()
    resolved_data_dir = data_dir.expanduser().resolve()
    resolved_config_dir = (resolved_repo_root / resolve_study_config_dir(config_dir, smoke=smoke)).resolve()
    study_mode = resolve_study_mode(smoke=smoke)
    runs_root = resolved_study_dir / "runs"
    summaries_root = resolved_study_dir / "summaries"

    plans: list[StudyRunPlan] = []
    for dataset in datasets:
        normalized_dataset = normalize_dataset_name(dataset)
        recipe_path = build_recipe_path(resolved_config_dir, normalized_dataset)
        recipe = load_recipe(recipe_path)
        config_name = str(recipe.values["config_name"])
        protocol_name = recipe.values.get("protocol_name")

        for seed in seeds:
            run_name = build_study_run_name(config_name, seed)
            train_run_dir = runs_root / normalized_dataset / "diffusion" / run_name
            checkpoint_path = train_run_dir / "checkpoints" / "best.pt"
            eval_run_name = f"eval_{run_name}"
            evaluation_dir = train_run_dir / "evaluations" / eval_run_name
            evaluation_metrics_path = evaluation_dir / "metrics.json"
            train_command = [
                sys.executable,
                "train.py",
                "--config",
                str(recipe_path),
                "--seed",
                str(seed),
                "--run-name",
                run_name,
                "--data-dir",
                str(resolved_data_dir),
                "--output-dir",
                str(runs_root),
            ]
            eval_command = [
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                str(checkpoint_path),
                "--mode",
                "evaluate",
                "--run-name",
                eval_run_name,
                "--data-dir",
                str(resolved_data_dir),
            ]
            if allow_model_download:
                eval_command.append("--allow-model-download")
            if smoke:
                eval_command.extend(SMOKE_EVAL_COMMAND_OVERRIDES)
            plans.append(
                StudyRunPlan(
                    study_mode=study_mode,
                    dataset=normalized_dataset,
                    seed=seed,
                    recipe_path=recipe_path,
                    config_name=config_name,
                    protocol_name=protocol_name,
                    study_dir=resolved_study_dir,
                    data_dir=resolved_data_dir,
                    runs_root=runs_root,
                    summaries_root=summaries_root,
                    run_name=run_name,
                    eval_run_name=eval_run_name,
                    train_run_dir=train_run_dir,
                    checkpoint_path=checkpoint_path,
                    evaluation_dir=evaluation_dir,
                    evaluation_metrics_path=evaluation_metrics_path,
                    train_command=train_command,
                    eval_command=eval_command,
                )
            )
    return plans


def registry_path(study_dir: Path) -> Path:
    return study_dir / "study_registry.json"


def _study_entry_id(plan: StudyRunPlan) -> str:
    return f"{plan.dataset}:{plan.config_name}:seed{plan.seed:03d}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(study_dir: Path) -> dict[str, Any]:
    path = registry_path(study_dir)
    if not path.exists():
        return {
            "study_dir": str(study_dir.resolve()),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "entries": {},
        }
    return _load_json(path)


def save_registry(study_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    payload["updated_at"] = _utc_now()
    manifest_paths = save_manifest_bundle(
        study_dir,
        basename="study_registry",
        title="Final Diffusion Study Registry",
        payload=payload,
    )
    return manifest_paths


def _infer_entry_study_mode(entry: dict[str, Any]) -> str:
    recipe_path = str(entry.get("recipe_path") or "").replace("\\", "/")
    config_name = str(entry.get("config_name") or "")
    if "/configs/diffusion/smoke/" in recipe_path or config_name.endswith("_smoke"):
        return "smoke"
    return "full"


def _ensure_study_mode_compatible(registry: dict[str, Any], *, plans: list[StudyRunPlan], study_dir: Path) -> None:
    planned_modes = {plan.study_mode for plan in plans}
    if not planned_modes:
        return
    if len(planned_modes) != 1:  # pragma: no cover - defensive
        raise ValueError(f"Study plans must agree on one mode, got {sorted(planned_modes)}.")
    planned_mode = next(iter(planned_modes))
    if planned_mode not in SUPPORTED_STUDY_MODES:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported study mode: {planned_mode}")

    existing_mode = registry.get("study_mode")
    if existing_mode in SUPPORTED_STUDY_MODES:
        if existing_mode != planned_mode:
            raise ValueError(
                f"Study directory {study_dir} is already initialized for {existing_mode} runs. "
                "Keep smoke and full studies in separate study directories."
            )
        registry["study_mode"] = existing_mode
        return

    entries = list((registry.get("entries") or {}).values())
    if not entries:
        registry["study_mode"] = planned_mode
        return

    inferred_modes = {_infer_entry_study_mode(entry) for entry in entries}
    if len(inferred_modes) != 1:
        raise ValueError(
            f"Study directory {study_dir} already contains mixed study outputs. "
            "Keep smoke and full studies in separate study directories."
        )
    inferred_mode = next(iter(inferred_modes))
    if inferred_mode != planned_mode:
        raise ValueError(
            f"Study directory {study_dir} already contains {inferred_mode} runs. "
            "Keep smoke and full studies in separate study directories."
        )
    registry["study_mode"] = inferred_mode


def _initial_registry_entry(plan: StudyRunPlan, git_commit: str | None) -> dict[str, Any]:
    return {
        "study_mode": plan.study_mode,
        "dataset": plan.dataset,
        "seed": plan.seed,
        "recipe_path": str(plan.recipe_path.resolve()),
        "config_name": plan.config_name,
        "protocol_name": plan.protocol_name,
        "study_dir": str(plan.study_dir.resolve()),
        "data_dir": str(plan.data_dir.resolve()),
        "run_name": plan.run_name,
        "eval_run_name": plan.eval_run_name,
        "train_run_dir": str(plan.train_run_dir.resolve()),
        "checkpoint_path": str(plan.checkpoint_path.resolve()),
        "evaluation_dir": str(plan.evaluation_dir.resolve()),
        "evaluation_metrics_path": str(plan.evaluation_metrics_path.resolve()),
        "train_command": plan.train_command,
        "eval_command": plan.eval_command,
        "git_commit": git_commit,
        "train_status": "pending",
        "eval_status": "pending",
        "train_started_at": None,
        "train_completed_at": None,
        "eval_started_at": None,
        "eval_completed_at": None,
        "train_manifest_path": None,
        "evaluation_manifest_path": None,
        "selected_checkpoint_path": None,
        "selected_evaluation_dir": None,
    }


def initialize_registry(
    study_dir: Path,
    plans: list[StudyRunPlan],
    *,
    git_commit: str | None,
) -> dict[str, Any]:
    registry = load_registry(study_dir)
    _ensure_study_mode_compatible(registry, plans=plans, study_dir=study_dir)
    entries = dict(registry.get("entries") or {})
    for plan in plans:
        entry_id = _study_entry_id(plan)
        entries.setdefault(entry_id, _initial_registry_entry(plan, git_commit))
    registry["entries"] = entries
    registry.setdefault("created_at", _utc_now())
    save_registry(study_dir, registry)
    return registry


def is_train_complete(plan: StudyRunPlan) -> bool:
    return plan.checkpoint_path.exists() and (plan.train_run_dir / "metrics.json").exists()


def is_eval_complete(plan: StudyRunPlan) -> bool:
    return plan.evaluation_metrics_path.exists()


def ensure_path_safe(plan: StudyRunPlan, *, phase: str, skip_existing: bool) -> None:
    """Prevent ambiguous reruns or accidental suffix-based output drift."""

    if phase == "train":
        if plan.train_run_dir.exists() and not is_train_complete(plan):
            raise FileExistsError(
                f"Train run directory already exists but is incomplete: {plan.train_run_dir}. "
                "Refusing to launch a second run into the same name."
            )
        if plan.train_run_dir.exists() and is_train_complete(plan) and not skip_existing:
            raise FileExistsError(
                f"Train run already exists at {plan.train_run_dir}. Use --skip-existing to reuse it."
            )
    elif phase == "eval":
        if plan.evaluation_dir.exists() and not is_eval_complete(plan):
            raise FileExistsError(
                f"Evaluation directory already exists but is incomplete: {plan.evaluation_dir}. "
                "Refusing to overwrite it."
            )
        if plan.evaluation_dir.exists() and is_eval_complete(plan) and not skip_existing:
            raise FileExistsError(
                f"Evaluation already exists at {plan.evaluation_dir}. Use --skip-existing to reuse it."
            )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported phase: {phase}")


Runner = Callable[[list[str], Path], None]


def subprocess_runner(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def update_registry_entry_from_outputs(
    entry: dict[str, Any],
    *,
    plan: StudyRunPlan,
) -> None:
    train_manifest_path = plan.train_run_dir / "run_manifest.json"
    if train_manifest_path.exists():
        entry["train_manifest_path"] = str(train_manifest_path.resolve())
        entry["selected_checkpoint_path"] = str(plan.checkpoint_path.resolve())
    evaluation_manifest_path = plan.evaluation_dir / "evaluation_manifest.json"
    if evaluation_manifest_path.exists():
        entry["evaluation_manifest_path"] = str(evaluation_manifest_path.resolve())
        entry["selected_evaluation_dir"] = str(plan.evaluation_dir.resolve())


def execute_parity_suite(
    *,
    study_dir: Path,
    plans: list[StudyRunPlan],
    phase: str,
    skip_existing: bool,
    runner: Runner = subprocess_runner,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run train/eval phases via the existing entrypoints and update the study registry."""

    if phase not in SUPPORTED_PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    resolved_repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    git_commit = detect_git_commit(resolved_repo_root)
    registry = initialize_registry(study_dir, plans, git_commit=git_commit)
    entries = registry["entries"]

    for plan in plans:
        entry = entries[_study_entry_id(plan)]

        if phase in {"train", "both"}:
            ensure_path_safe(plan, phase="train", skip_existing=skip_existing)
            if is_train_complete(plan) and skip_existing:
                entry["train_status"] = "skipped_existing"
                entry["train_completed_at"] = entry.get("train_completed_at") or _utc_now()
                update_registry_entry_from_outputs(entry, plan=plan)
            else:
                entry["train_status"] = "running"
                entry["train_started_at"] = _utc_now()
                save_registry(study_dir, registry)
                runner(plan.train_command, resolved_repo_root)
                entry["train_status"] = "completed"
                entry["train_completed_at"] = _utc_now()
                update_registry_entry_from_outputs(entry, plan=plan)
                save_registry(study_dir, registry)

        if phase in {"eval", "both"}:
            if not is_train_complete(plan):
                raise FileNotFoundError(
                    f"Cannot evaluate without a completed training run at {plan.checkpoint_path}."
                )
            ensure_path_safe(plan, phase="eval", skip_existing=skip_existing)
            if is_eval_complete(plan) and skip_existing:
                entry["eval_status"] = "skipped_existing"
                entry["eval_completed_at"] = entry.get("eval_completed_at") or _utc_now()
                update_registry_entry_from_outputs(entry, plan=plan)
            else:
                entry["eval_status"] = "running"
                entry["eval_started_at"] = _utc_now()
                save_registry(study_dir, registry)
                runner(plan.eval_command, resolved_repo_root)
                entry["eval_status"] = "completed"
                entry["eval_completed_at"] = _utc_now()
                update_registry_entry_from_outputs(entry, plan=plan)
                save_registry(study_dir, registry)

    save_registry(study_dir, registry)
    return registry


def write_study_plan_files(
    *,
    study_dir: Path,
    plans: list[StudyRunPlan],
    phase: str,
) -> dict[str, str]:
    """Export shell and Slurm-friendly command plans for the suite."""

    if phase not in SUPPORTED_PHASES:
        raise ValueError(f"Unsupported phase: {phase}")

    commands: list[list[str]] = []
    for plan in plans:
        if phase in {"train", "both"}:
            commands.append(plan.train_command)
        if phase in {"eval", "both"}:
            commands.append(plan.eval_command)

    plan_dir = study_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    commands_path = plan_dir / "commands.txt"
    commands_path.write_text("\n".join(shlex.join(command) for command in commands) + "\n", encoding="utf-8")

    shell_path = plan_dir / "planned_commands.sh"
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    shell_lines.extend(shlex.join(command) for command in commands)
    shell_path.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")

    slurm_path = plan_dir / "planned_array.slurm"
    slurm_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=diffusion-study",
        f"#SBATCH --array=0-{max(len(commands) - 1, 0)}",
        "#SBATCH --output=logs/diffusion-study-%A_%a.out",
        "",
        f"mapfile -t COMMANDS < {commands_path}",
        'eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"',
        "",
    ]
    slurm_path.write_text("\n".join(slurm_lines), encoding="utf-8")

    manifest_payload = {
        "phase": phase,
        "command_count": len(commands),
        "commands": [shlex.join(command) for command in commands],
        "plans": [asdict(plan) for plan in plans],
    }
    manifest_paths = save_manifest_bundle(
        plan_dir,
        basename="study_plan",
        title="Final Diffusion Study Plan",
        payload=manifest_payload,
    )
    return {
        **manifest_paths,
        "commands_txt": str(commands_path.resolve()),
        "shell_script": str(shell_path.resolve()),
        "slurm_script": str(slurm_path.resolve()),
    }


def _safe_metric_value(row: dict[str, Any], metric_name: str) -> float | None:
    value = row.get(metric_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_best_and_median_runs(
    rows: list[dict[str, Any]],
    *,
    primary_metric: str = "fid",
) -> dict[str, dict[str, dict[str, Any]]]:
    """Select the best and median run per dataset using the chosen metric."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metric_value = _safe_metric_value(row, primary_metric)
        if metric_value is None:
            continue
        grouped.setdefault(str(row["dataset"]), []).append(row)

    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, dataset_rows in grouped.items():
        ordered = sorted(
            dataset_rows,
            key=lambda row: (
                _safe_metric_value(row, primary_metric),
                int(str(row.get("seed") or 0)) if row.get("seed") is not None else 0,
                str(row.get("evaluation_dir") or ""),
            ),
        )
        best = ordered[0]
        median = ordered[len(ordered) // 2]
        selections[dataset] = {
            "best": best,
            "median": median,
            "selection_rule": {
                "primary_metric": primary_metric,
                "best_rule": f"minimum {primary_metric}",
                "median_rule": f"sorted by {primary_metric}, choose index floor(n/2)",
            },
        }
    return selections


def summarize_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute mean/std summaries across seeds for each dataset."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for dataset, dataset_rows in sorted(grouped.items()):
        metric_names = ("fid", "inception_score_mean", "lpips_diversity", "psnr", "ssim")
        summary: dict[str, Any] = {
            "dataset": dataset,
            "run_count": len(dataset_rows),
        }
        for metric_name in metric_names:
            values = [value for value in (_safe_metric_value(row, metric_name) for row in dataset_rows) if value is not None]
            if values:
                summary[f"{metric_name}_mean"] = statistics.fmean(values)
                summary[f"{metric_name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            else:
                summary[f"{metric_name}_mean"] = None
                summary[f"{metric_name}_std"] = None
        summaries.append(summary)
    return summaries


def build_per_run_rows(
    registry: dict[str, Any],
    evaluation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge registry bookkeeping with evaluation metrics for per-run summaries."""

    evaluation_by_checkpoint = {
        row.get("checkpoint_path"): row
        for row in evaluation_rows
        if row.get("checkpoint_path") is not None
    }
    per_run_rows: list[dict[str, Any]] = []
    for entry_id, entry in sorted((registry.get("entries") or {}).items()):
        linked_row = evaluation_by_checkpoint.get(entry.get("checkpoint_path"))
        merged_row = {
            "entry_id": entry_id,
            **entry,
        }
        if linked_row is not None:
            merged_row.update({f"eval_{key}": value for key, value in linked_row.items()})
            for metric_name in ("fid", "inception_score_mean", "lpips_diversity", "psnr", "ssim"):
                merged_row[metric_name] = linked_row.get(metric_name)
        per_run_rows.append(merged_row)
    return per_run_rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_final_study_summaries(
    *,
    study_dir: Path,
    inputs: list[Path] | None = None,
) -> dict[str, Any]:
    """Build per-run, per-dataset, and best-run summaries for the final study."""

    resolved_study_dir = study_dir.expanduser().resolve()
    runs_root = resolved_study_dir / "runs"
    metric_inputs = list(inputs or [runs_root])
    metric_files = discover_metric_files(metric_inputs)
    evaluation_rows = aggregate_evaluation_results(metric_files)
    aggregate_outputs = save_aggregate_outputs(resolved_study_dir / "summaries" / "aggregate", evaluation_rows)

    registry = load_registry(resolved_study_dir)
    per_run_rows = build_per_run_rows(registry, evaluation_rows)
    per_dataset_rows = summarize_by_dataset(evaluation_rows)
    selection_payload = select_best_and_median_runs(evaluation_rows)
    for summary_row in per_dataset_rows:
        selection = selection_payload.get(summary_row["dataset"])
        if selection is None:
            continue
        best = selection["best"]
        summary_row["best_fid"] = best.get("fid")
        summary_row["best_evaluation_dir"] = best.get("evaluation_dir")
        summary_row["best_generated_sample_grid"] = best.get("generated_sample_grid")
        summary_row["best_generated_samples"] = best.get("generated_samples") or best.get("generated_sample_grid")
        summary_row["best_cfg_comparison_grid"] = best.get("cfg_comparison_grid")
        summary_row["best_diffusion_snapshots"] = best.get("diffusion_snapshots") or best.get("reverse_process_snapshots")
        summary_row["best_reverse_process_snapshots"] = best.get("diffusion_snapshots") or best.get("reverse_process_snapshots")
        summary_row["best_nearest_neighbor_grid"] = best.get("nearest_neighbor_grid")
        summary_row["best_reconstructions"] = best.get("reconstructions") or best.get("reconstruction_preview")

    summaries_root = resolved_study_dir / "summaries"
    per_run_dir = summaries_root / "per_run"
    per_dataset_dir = summaries_root / "per_dataset"
    selections_dir = summaries_root / "selections"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    selections_dir.mkdir(parents=True, exist_ok=True)

    _save_csv(per_run_dir / "per_run_summary.csv", per_run_rows)
    per_run_manifest = save_manifest_bundle(
        per_run_dir,
        basename="per_run_summary",
        title="Per-Run Final Diffusion Study Summary",
        payload={"rows": per_run_rows},
    )

    _save_csv(per_dataset_dir / "per_dataset_summary.csv", per_dataset_rows)
    per_dataset_manifest = save_manifest_bundle(
        per_dataset_dir,
        basename="per_dataset_summary",
        title="Per-Dataset Final Diffusion Study Summary",
        payload={"rows": per_dataset_rows},
    )

    best_rows = [
        {
            "dataset": dataset,
            "best": selection["best"],
            "median": selection["median"],
            "selection_rule": selection["selection_rule"],
        }
        for dataset, selection in sorted(selection_payload.items())
    ]
    best_manifest = save_manifest_bundle(
        selections_dir,
        basename="best_runs",
        title="Best And Median Runs By Dataset",
        payload={"rows": best_rows},
    )
    selection_csv_rows = []
    for dataset, selection in sorted(selection_payload.items()):
        for label in ("best", "median"):
            row = {"dataset": dataset, "selection": label, **selection[label]}
            selection_csv_rows.append(row)
    _save_csv(selections_dir / "best_runs.csv", selection_csv_rows)

    report_lines = [
        "# Final Diffusion Study Summary",
        "",
        f"- `study_dir`: `{resolved_study_dir}`",
        f"- `run_count`: `{len(per_run_rows)}`",
        f"- `evaluation_count`: `{len(evaluation_rows)}`",
        "",
        "## Dataset Defaults",
    ]
    dataset_defaults: dict[str, dict[str, Any]] = {}
    for row in evaluation_rows:
        dataset = str(row.get("dataset") or "")
        if dataset and dataset not in dataset_defaults:
            dataset_defaults[dataset] = row
    for dataset, row in sorted(dataset_defaults.items()):
        report_lines.append(
            f"- `{dataset}`: backbone=`{row.get('diffusion_backbone')}` "
            f"shape=`{row.get('diffusion_channels')}x{row.get('image_size')}x{row.get('image_size')}` "
            f"preproc=`{row.get('diffusion_preprocessing')}`"
        )
    report_lines.extend(
        [
            "",
            "## Per-Dataset Means",
        ]
    )
    for row in per_dataset_rows:
        report_lines.append(
            f"- `{row['dataset']}`: FID mean=`{row['fid_mean']}` std=`{row['fid_std']}` "
            f"IS mean=`{row['inception_score_mean_mean']}`"
        )
    report_lines.append("")
    report_lines.append("## Best Runs")
    for dataset, selection in sorted(selection_payload.items()):
        best = selection["best"]
        report_lines.append(
            f"- `{dataset}`: best FID run=`{best.get('config_name')}` metrics=`{best.get('metrics_path')}`"
        )
    report_path = summaries_root / "final_study_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "evaluation_rows": evaluation_rows,
        "per_run_rows": per_run_rows,
        "per_dataset_rows": per_dataset_rows,
        "selection_payload": selection_payload,
        "aggregate_outputs": aggregate_outputs,
        "per_run_manifest": per_run_manifest,
        "per_dataset_manifest": per_dataset_manifest,
        "best_manifest": best_manifest,
        "report_path": str(report_path.resolve()),
    }
