from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from diffusion.reporting import save_manifest_bundle


def discover_metric_files(inputs: list[Path]) -> list[Path]:
    """Find evaluation metrics files under explicit paths or search roots."""

    discovered: list[Path] = []
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            discovered.append(path)
            continue

        discovered.extend(
            sorted(
                candidate
                for candidate in path.rglob("metrics.json")
                if "evaluations" in candidate.parts
            )
        )
    return list(dict.fromkeys(discovered))


def load_metric_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "generative_metrics" not in payload and "paired_metrics" not in payload:
        raise ValueError(f"{path} does not look like an evaluation metrics payload.")
    return payload


def build_aggregate_row(payload: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    generative_metrics = payload.get("generative_metrics") or {}
    paired_metrics = payload.get("paired_metrics") or {}
    artifacts = payload.get("artifacts") or {}

    return {
        "dataset": payload.get("dataset"),
        "config_name": payload.get("config_name"),
        "protocol_name": payload.get("protocol_name"),
        "dataset_variant": payload.get("dataset_variant"),
        "seed": payload.get("seed"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "evaluation_dir": payload.get("evaluation_dir"),
        "metrics_path": str(metrics_path.resolve()),
        "image_size": payload.get("image_size"),
        "diffusion_channels": payload.get("diffusion_channels"),
        "diffusion_preprocessing": payload.get("diffusion_preprocessing"),
        "diffusion_backbone": payload.get("diffusion_backbone"),
        "prediction_type": payload.get("prediction_type"),
        "sampler": payload.get("sampler"),
        "sampling_steps": payload.get("sampling_steps"),
        "guidance_scale": payload.get("guidance_scale"),
        "model_parameters": payload.get("model_parameters"),
        "fid": generative_metrics.get("fid"),
        "inception_score_mean": generative_metrics.get("inception_score_mean"),
        "inception_score_std": generative_metrics.get("inception_score_std"),
        "lpips_diversity": generative_metrics.get("lpips_diversity"),
        "psnr": paired_metrics.get("psnr"),
        "ssim": paired_metrics.get("ssim"),
        "generated_sample_grid": artifacts.get("generated_sample_grid"),
        "cfg_comparison_grid": artifacts.get("cfg_comparison_grid"),
        "reverse_process_snapshots": artifacts.get("reverse_process_snapshots"),
        "nearest_neighbor_grid": artifacts.get("nearest_neighbor_grid"),
    }


def aggregate_evaluation_results(metric_files: list[Path]) -> list[dict[str, Any]]:
    """Load and normalize evaluation payloads into a comparison table."""

    rows = [build_aggregate_row(load_metric_payload(path), path) for path in metric_files]
    return sorted(rows, key=lambda row: (row.get("protocol_name") or "", row.get("dataset") or "", row.get("config_name") or ""))


def save_aggregate_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Persist combined comparison results as CSV, JSON, and Markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_count": len(rows),
        "rows": rows,
    }
    manifest_paths = save_manifest_bundle(
        output_dir,
        basename="aggregated_results",
        title="Aggregated Diffusion Results",
        payload=payload,
    )

    csv_path = output_dir / "aggregated_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    markdown_path = output_dir / "comparison_table.md"
    if rows:
        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        markdown_path.write_text("# Aggregated Diffusion Results\n\nNo rows found.\n", encoding="utf-8")

    return {
        **manifest_paths,
        "csv": str(csv_path.resolve()),
        "markdown_table": str(markdown_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed diffusion evaluation results into one comparison table.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more evaluation metrics.json files or directories to search recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the combined report should be written.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    metric_files = discover_metric_files(list(args.inputs))
    rows = aggregate_evaluation_results(metric_files)
    output_paths = save_aggregate_outputs(args.output_dir, rows)
    print("Aggregated rows:")
    print(len(rows))
    print("Output directory:")
    print(args.output_dir.resolve())
    print("CSV:")
    print(output_paths["csv"])


if __name__ == "__main__":
    main()
