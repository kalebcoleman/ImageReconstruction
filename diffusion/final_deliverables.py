from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

from diffusion.parity_study import generate_final_study_summaries
from diffusion.reporting import save_manifest_bundle


ARTIFACT_EXPORT_NAMES: dict[str, str] = {
    "generated_sample_grid": "generated_samples",
    "cfg_comparison_grid": "cfg_comparison",
    "diffusion_snapshots": "diffusion_snapshots",
    "reconstructions": "reconstructions",
    "nearest_neighbor_grid": "nearest_neighbors",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_rows_bundle(output_dir: Path, *, basename: str, title: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    payload = {"rows": rows, "row_count": len(rows)}
    manifest_paths = save_manifest_bundle(output_dir, basename=basename, title=title, payload=payload)
    csv_path = output_dir / f"{basename}.csv"
    _write_csv(csv_path, rows)
    return {
        **manifest_paths,
        "csv": str(csv_path.resolve()),
    }


def build_best_run_rows(selection_payload: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten the best-run selection payload into one row per dataset."""

    rows: list[dict[str, Any]] = []
    for dataset, selection in sorted(selection_payload.items()):
        best = selection["best"]
        rows.append(
            {
                "dataset": dataset,
                "selection_metric": selection["selection_rule"]["primary_metric"],
                "selection_rule": selection["selection_rule"]["best_rule"],
                "seed": best.get("seed"),
                "config_name": best.get("config_name"),
                "protocol_name": best.get("protocol_name"),
                "fid": best.get("fid"),
                "inception_score_mean": best.get("inception_score_mean"),
                "inception_score_std": best.get("inception_score_std"),
                "lpips_diversity": best.get("lpips_diversity"),
                "psnr": best.get("psnr"),
                "ssim": best.get("ssim"),
                "checkpoint_path": best.get("checkpoint_path"),
                "evaluation_dir": best.get("evaluation_dir"),
                "metrics_path": best.get("metrics_path"),
                "generated_sample_grid": best.get("generated_sample_grid"),
                "generated_samples": best.get("generated_samples") or best.get("generated_sample_grid"),
                "cfg_comparison_grid": best.get("cfg_comparison_grid"),
                "diffusion_snapshots": best.get("diffusion_snapshots") or best.get("reverse_process_snapshots"),
                "reverse_process_snapshots": best.get("diffusion_snapshots") or best.get("reverse_process_snapshots"),
                "nearest_neighbor_grid": best.get("nearest_neighbor_grid"),
                "reconstructions": best.get("reconstructions") or best.get("reconstruction_preview"),
                "reconstruction_preview": best.get("reconstructions") or best.get("reconstruction_preview"),
            }
        )
    return rows


def build_main_results_rows(best_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the concise main-results table used in the final report bundle."""

    return [
        {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "fid": row["fid"],
            "inception_score_mean": row["inception_score_mean"],
            "inception_score_std": row["inception_score_std"],
            "lpips_diversity": row["lpips_diversity"],
            "psnr_aux": row["psnr"],
            "ssim_aux": row["ssim"],
        }
        for row in best_run_rows
    ]


def build_artifact_index(best_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index the presentation-relevant artifacts from the best run per dataset."""

    rows: list[dict[str, Any]] = []
    for row in best_run_rows:
        dataset = row["dataset"]
        for artifact_key, export_stem in ARTIFACT_EXPORT_NAMES.items():
            source_path = row.get(artifact_key)
            rows.append(
                {
                    "dataset": dataset,
                    "artifact_key": artifact_key,
                    "export_stem": export_stem,
                    "source_path": source_path,
                    "exists": bool(source_path) and Path(source_path).exists(),
                }
            )
    return rows


def export_best_artifacts(
    artifact_index_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Copy the highest-value figures into one stable final-deliverables folder."""

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_rows: list[dict[str, Any]] = []
    for row in artifact_index_rows:
        dataset = row["dataset"]
        source_path_value = row.get("source_path")
        exported_path = None
        if source_path_value:
            source_path = Path(str(source_path_value))
            if source_path.exists():
                target_path = output_dir / f"{dataset}_{row['export_stem']}{source_path.suffix or '.png'}"
                shutil.copy2(source_path, target_path)
                exported_path = str(target_path.resolve())
        exported_rows.append(
            {
                **row,
                "exported_path": exported_path,
            }
        )
    return exported_rows


def build_analysis_summary_markdown(
    *,
    study_dir: Path,
    main_results_rows: list[dict[str, Any]],
    mean_std_rows: list[dict[str, Any]],
    best_run_rows: list[dict[str, Any]],
) -> str:
    """Generate a report-ready analysis scaffold from completed study outputs."""

    lines = [
        "# Final Study Analysis Summary",
        "",
        f"- `study_dir`: `{study_dir.resolve()}`",
        "",
        "## Experimental Setup",
        "- Dataset-appropriate diffusion study on MNIST, FashionMNIST, and CIFAR10.",
        "- MNIST and FashionMNIST use the legacy 28x28 grayscale diffusion baseline by default.",
        "- CIFAR10 uses the ADM 32x32 RGB diffusion path with native-image preprocessing.",
        "- Checkpoint-only evaluation executed through `evaluate.py`.",
        "",
        "## Study Defaults",
        "- `mnist`: legacy backbone, `28x28`, `1` channel, grayscale preprocessing.",
        "- `fashion`: legacy backbone, `28x28`, `1` channel, grayscale preprocessing.",
        "- `cifar10`: ADM backbone, `32x32`, `3` channels, native-image preprocessing.",
        "",
        "## Main Findings By Dataset",
    ]
    for row in main_results_rows:
        lines.append(
            f"- `{row['dataset']}`: best-seed FID=`{row['fid']}`, "
            f"IS=`{row['inception_score_mean']}` +/- `{row['inception_score_std']}`, "
            f"LPIPS diversity=`{row['lpips_diversity']}`. "
            "[Add interpretation here.]"
        )

    lines.extend(
        [
            "",
            "## Mean / Std Across Seeds",
        ]
    )
    for row in mean_std_rows:
        lines.append(
            f"- `{row['dataset']}`: FID mean=`{row['fid_mean']}` std=`{row['fid_std']}`; "
            f"IS mean=`{row['inception_score_mean_mean']}` std=`{row['inception_score_mean_std']}`. "
            "[Add stability interpretation here.]"
        )

    lines.extend(
        [
            "",
            "## CFG Observations",
            "- Review the exported CFG comparison grids for CIFAR10 or any other explicitly conditional ADM runs.",
            "- Legacy grayscale checkpoints keep `guidance_scale=1.0` by default, so CFG is not a main comparison axis there.",
            "- [Add qualitative CFG observations here.]",
            "",
            "## DDPM vs DDIM Observations",
            "- The default study recipes use DDIM for evaluation artifacts by default.",
            "- If DDPM comparisons were run separately, add them here with explicit commands and artifacts.",
            "",
            "## Limitations",
            "- Auxiliary paired metrics (`PSNR`, `SSIM`) are not primary generative metrics.",
            "- The study is limited to MNIST, FashionMNIST, and CIFAR10 for the final course deliverable.",
            "- [Add dataset- or compute-specific limitations here.]",
            "",
            "## Future Work",
            "- Explore broader ablations only after preserving the dataset-appropriate defaults as a baseline.",
            "- [Add post-course extensions here.]",
            "",
            "## Best Artifact References",
        ]
    )
    for row in best_run_rows:
        lines.append(
            f"- `{row['dataset']}`: samples=`{row['generated_sample_grid']}`, "
            f"cfg=`{row['cfg_comparison_grid']}`, "
            f"snapshots=`{row['diffusion_snapshots']}`, "
            f"reconstructions=`{row['reconstructions']}`, "
            f"nearest=`{row['nearest_neighbor_grid']}`"
        )
    lines.append("")
    return "\n".join(lines)


def build_presentation_index_markdown(exported_artifact_rows: list[dict[str, Any]]) -> str:
    """Create a presentation-oriented figure reference sheet."""

    lines = [
        "# Presentation Figure Index",
        "",
        "Use the following exported artifacts for the presentation deck.",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in exported_artifact_rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    for dataset, rows in sorted(grouped.items()):
        lines.append(f"## {dataset}")
        for row in rows:
            lines.append(
                f"- `{row['artifact_key']}`: source=`{row['source_path']}` exported=`{row['exported_path']}`"
            )
        lines.append("")
    return "\n".join(lines)


def export_final_deliverables(
    *,
    study_dir: Path,
    output_dir: Path | None = None,
    inputs: list[Path] | None = None,
) -> dict[str, Any]:
    """Export the polished final deliverables bundle for the completed study."""

    resolved_study_dir = study_dir.expanduser().resolve()
    resolved_output_dir = (output_dir or (resolved_study_dir / "deliverables")).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = generate_final_study_summaries(study_dir=resolved_study_dir, inputs=inputs)
    best_run_rows = build_best_run_rows(summary_payload["selection_payload"])
    main_results_rows = build_main_results_rows(best_run_rows)
    mean_std_rows = summary_payload["per_dataset_rows"]
    artifact_index_rows = build_artifact_index(best_run_rows)
    exported_artifact_rows = export_best_artifacts(
        artifact_index_rows,
        output_dir=resolved_output_dir / "figures",
    )

    tables_dir = resolved_output_dir / "tables"
    figures_dir = resolved_output_dir / "figures"
    summaries_dir = resolved_output_dir / "summaries"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    main_results_paths = _save_rows_bundle(
        tables_dir,
        basename="main_results_table",
        title="Main Results Table",
        rows=main_results_rows,
    )
    mean_std_paths = _save_rows_bundle(
        tables_dir,
        basename="mean_std_table",
        title="Mean / Std Across Seeds",
        rows=mean_std_rows,
    )
    best_runs_paths = _save_rows_bundle(
        tables_dir,
        basename="best_runs_table",
        title="Best Runs By Dataset",
        rows=best_run_rows,
    )
    artifact_index_paths = _save_rows_bundle(
        tables_dir,
        basename="artifact_index",
        title="Artifact Index",
        rows=exported_artifact_rows,
    )

    analysis_summary_path = summaries_dir / "analysis_summary.md"
    analysis_summary_path.write_text(
        build_analysis_summary_markdown(
            study_dir=resolved_study_dir,
            main_results_rows=main_results_rows,
            mean_std_rows=mean_std_rows,
            best_run_rows=best_run_rows,
        ),
        encoding="utf-8",
    )
    project_report_summary_path = summaries_dir / "project_report_summary.md"
    project_report_summary_path.write_text(
        build_analysis_summary_markdown(
            study_dir=resolved_study_dir,
            main_results_rows=main_results_rows,
            mean_std_rows=mean_std_rows,
            best_run_rows=best_run_rows,
        ),
        encoding="utf-8",
    )
    presentation_index_path = summaries_dir / "presentation_figure_index.md"
    presentation_index_path.write_text(
        build_presentation_index_markdown(exported_artifact_rows),
        encoding="utf-8",
    )

    bundle_payload = {
        "study_dir": str(resolved_study_dir),
        "output_dir": str(resolved_output_dir),
        "main_results_table": main_results_paths,
        "mean_std_table": mean_std_paths,
        "best_runs_table": best_runs_paths,
        "artifact_index": artifact_index_paths,
        "analysis_summary_path": str(analysis_summary_path.resolve()),
        "project_report_summary_path": str(project_report_summary_path.resolve()),
        "presentation_figure_index_path": str(presentation_index_path.resolve()),
        "exported_artifacts": exported_artifact_rows,
        "source_summary_report": summary_payload["report_path"],
    }
    bundle_manifest_paths = save_manifest_bundle(
        resolved_output_dir,
        basename="deliverables_bundle",
        title="Final Deliverables Bundle",
        payload=bundle_payload,
    )
    bundle_payload["bundle_manifest"] = bundle_manifest_paths
    (resolved_output_dir / "deliverables_bundle.json").write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle_payload
