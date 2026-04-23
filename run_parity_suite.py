"""Compatibility-named CLI wrapper for the dataset-appropriate final diffusion study."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusion.data import normalize_dataset_name
from diffusion.final_deliverables import (
    build_artifact_index,
    build_best_run_rows,
    export_best_artifacts,
    export_final_deliverables,
)
from diffusion.parity_study import (
    DEFAULT_STUDY_SEEDS,
    DEFAULT_SMOKE_STUDY_SEEDS,
    FINAL_STUDY_DATASETS,
    build_study_plans,
    execute_parity_suite,
    generate_final_study_summaries,
    select_best_and_median_runs,
    write_study_plan_files,
)


def _parse_datasets(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return FINAL_STUDY_DATASETS
    return tuple(normalize_dataset_name(value) for value in values)


def _parse_seeds(values: list[int] | None, *, smoke: bool) -> tuple[int, ...]:
    if not values:
        if smoke:
            return DEFAULT_SMOKE_STUDY_SEEDS
        return DEFAULT_STUDY_SEEDS
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final diffusion study for MNIST, FashionMNIST, and CIFAR10."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--study-dir", type=Path, required=True, help="Root directory for the final-study outputs and registry.")
    common_run.add_argument("--data-dir", type=Path, required=True, help="Dataset root shared by the final-study recipes.")
    common_run.add_argument("--datasets", nargs="*", default=None, help="Datasets to include. Defaults to the final-study set: mnist fashion cifar10.")
    common_run.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds to run. Defaults to 1 2 3.")
    common_run.add_argument("--config-dir", type=Path, default=Path("configs/diffusion"))
    common_run.add_argument("--smoke", action="store_true", help="Use the lightweight smoke-test recipes and default to seed 1 when --seeds is omitted.")
    common_run.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to train.py so datasets can be downloaded when missing. Defaults to false for cluster safety.",
    )
    common_run.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Pass through to evaluate.py so missing pretrained metric-model weights can be downloaded during evaluation.",
    )

    run_parser = subparsers.add_parser("run", parents=[common_run], help="Launch train/eval phases for the final diffusion study.")
    run_parser.add_argument("--phase", choices=("train", "eval", "both"), default="both")
    run_parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    run_parser.add_argument(
        "--clear-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete incomplete train/eval outputs for the planned deterministic run names before rerunning.",
    )
    run_parser.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True, help="Regenerate study summaries after execution.")

    plan_parser = subparsers.add_parser("plan", parents=[common_run], help="Write shell and Slurm command plans without executing them.")
    plan_parser.add_argument("--phase", choices=("train", "eval", "both"), default="both")

    summarize_parser = subparsers.add_parser("summarize", help="Regenerate per-run, per-dataset, and cross-dataset study summaries.")
    summarize_parser.add_argument("--study-dir", type=Path, required=True)
    summarize_parser.add_argument("inputs", nargs="*", type=Path, help="Optional evaluation metrics files or directories. Defaults to <study-dir>/runs.")

    best_parser = subparsers.add_parser("select-best", help="Select best and median runs per dataset by FID.")
    best_parser.add_argument("--study-dir", type=Path, required=True)
    best_parser.add_argument("inputs", nargs="*", type=Path, help="Optional evaluation metrics files or directories. Defaults to <study-dir>/runs.")

    deliverables_parser = subparsers.add_parser("deliverables", help="Generate the polished final deliverables bundle.")
    deliverables_parser.add_argument("--study-dir", type=Path, required=True)
    deliverables_parser.add_argument("--output-dir", type=Path, default=None, help="Optional export directory. Defaults to <study-dir>/deliverables.")
    deliverables_parser.add_argument(
        "--presentation-copies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create nearest-neighbor upscaled *_presentation.png copies of exported figure artifacts.",
    )
    deliverables_parser.add_argument(
        "--presentation-scale",
        type=int,
        default=4,
        help="Integer scale factor for *_presentation.png copies. Defaults to 4.",
    )
    deliverables_parser.add_argument("inputs", nargs="*", type=Path, help="Optional evaluation metrics files or directories. Defaults to <study-dir>/runs.")

    export_parser = subparsers.add_parser("export-best-artifacts", help="Copy the best artifacts per dataset into one stable folder.")
    export_parser.add_argument("--study-dir", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, default=None, help="Optional artifact export directory. Defaults to <study-dir>/deliverables/figures.")
    export_parser.add_argument(
        "--presentation-copies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create nearest-neighbor upscaled *_presentation.png copies of exported figure artifacts.",
    )
    export_parser.add_argument(
        "--presentation-scale",
        type=int,
        default=4,
        help="Integer scale factor for *_presentation.png copies. Defaults to 4.",
    )
    export_parser.add_argument("inputs", nargs="*", type=Path, help="Optional evaluation metrics files or directories. Defaults to <study-dir>/runs.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"deliverables", "export-best-artifacts"} and args.presentation_scale < 1:
        parser.error("--presentation-scale must be at least 1")

    if args.command in {"run", "plan"}:
        datasets = _parse_datasets(args.datasets)
        seeds = _parse_seeds(args.seeds, smoke=args.smoke)
        plans = build_study_plans(
            study_dir=args.study_dir,
            data_dir=args.data_dir,
            datasets=datasets,
            seeds=seeds,
            config_dir=args.config_dir,
            smoke=args.smoke,
            download=args.download,
            allow_model_download=args.allow_model_download,
        )

        if args.command == "plan":
            outputs = write_study_plan_files(
                study_dir=args.study_dir.expanduser().resolve(),
                plans=plans,
                phase=args.phase,
            )
            print("Planned commands:")
            print(outputs["commands_txt"])
            print("Shell script:")
            print(outputs["shell_script"])
            print("Slurm script:")
            print(outputs["slurm_script"])
            return

        registry = execute_parity_suite(
            study_dir=args.study_dir.expanduser().resolve(),
            plans=plans,
            phase=args.phase,
            skip_existing=args.skip_existing,
            clear_incomplete=args.clear_incomplete,
        )
        print("Registry:")
        print((args.study_dir.expanduser().resolve() / "study_registry.json").resolve())
        print("Planned entries:")
        print(len(registry.get("entries") or {}))
        if args.summarize:
            summary_payload = generate_final_study_summaries(study_dir=args.study_dir)
            print("Summary report:")
            print(summary_payload["report_path"])
        return

    if args.command == "summarize":
        summary_payload = generate_final_study_summaries(
            study_dir=args.study_dir,
            inputs=list(args.inputs) if args.inputs else None,
        )
        print("Summary report:")
        print(summary_payload["report_path"])
        return

    if args.command == "select-best":
        summary_payload = generate_final_study_summaries(
            study_dir=args.study_dir,
            inputs=list(args.inputs) if args.inputs else None,
        )
        selections = select_best_and_median_runs(summary_payload["evaluation_rows"])
        for dataset, selection in sorted(selections.items()):
            best = selection["best"]
            median = selection["median"]
            print(f"{dataset}:")
            print(f"  best_fid={best.get('fid')} metrics={best.get('metrics_path')}")
            print(f"  median_fid={median.get('fid')} metrics={median.get('metrics_path')}")
        return

    if args.command == "deliverables":
        payload = export_final_deliverables(
            study_dir=args.study_dir,
            output_dir=args.output_dir,
            inputs=list(args.inputs) if args.inputs else None,
            presentation_copies=args.presentation_copies,
            presentation_scale=args.presentation_scale,
        )
        print("Deliverables bundle:")
        print(payload["bundle_manifest"]["json"])
        print("Project summary:")
        print(payload["project_report_summary_path"])
        return

    if args.command == "export-best-artifacts":
        summary_payload = generate_final_study_summaries(
            study_dir=args.study_dir,
            inputs=list(args.inputs) if args.inputs else None,
        )
        best_run_rows = build_best_run_rows(summary_payload["selection_payload"])
        artifact_index_rows = build_artifact_index(best_run_rows)
        artifact_rows = export_best_artifacts(
            artifact_index_rows,
            output_dir=(args.output_dir or (args.study_dir / "deliverables" / "figures")).expanduser().resolve(),
            presentation_copies=args.presentation_copies,
            presentation_scale=args.presentation_scale,
        )
        print("Exported artifacts:")
        for row in artifact_rows:
            if row.get("exported_path"):
                print(row["exported_path"])
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
