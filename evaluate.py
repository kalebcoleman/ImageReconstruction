from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion.eval_pipeline import CheckpointEvaluationConfig, run_checkpoint_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or sample from a trained diffusion checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a diffusion checkpoint such as checkpoints/best.pt.")
    parser.add_argument("--mode", choices=("evaluate", "sample"), default="evaluate")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the evaluation output root. Defaults to <run_dir>/evaluations/.")
    parser.add_argument("--run-name", default=None, help="Optional evaluation run name.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--num-generated-samples", type=int, default=None, help="Number of generated samples used for metrics or raw-image export. Defaults to the checkpoint recipe when available.")
    parser.add_argument("--eval-batch-size", "--batch-size", dest="eval_batch_size", type=int, default=None, help="Evaluation batch size. Defaults to the checkpoint recipe when available.")
    parser.add_argument("--num-workers", "--num-workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--ddim-eta", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--amp-dtype", choices=("auto", "none", "bf16", "fp16"), default=None)
    parser.add_argument("--artifact-sample-count", type=int, default=None, help="Number of images used in saved grids. Defaults to the checkpoint sample-count when available.")
    parser.add_argument(
        "--cfg-comparison-scales",
        nargs="*",
        type=float,
        default=None,
        help="Optional guidance scales for the CFG comparison artifact, for example '--cfg-comparison-scales 0 1 3 5'.",
    )
    parser.add_argument("--save-raw-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paired-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-stats-dir", type=Path, default=None)
    parser.add_argument("--allow-model-download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nearest-neighbor-count", type=int, default=8)
    parser.add_argument("--nearest-neighbor-reference-limit", type=int, default=10_000)
    parser.add_argument("--force-unconditional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-pair-count", type=int, default=128)
    args = parser.parse_args()

    if args.num_generated_samples is not None and args.num_generated_samples < 1:
        parser.error("--num-generated-samples must be at least 1")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("--sampling-steps must be at least 1")
    if args.ddim_eta is not None and args.ddim_eta < 0.0:
        parser.error("--ddim-eta must be non-negative")
    if args.guidance_scale is not None and args.guidance_scale < 0.0:
        parser.error("--guidance-scale must be non-negative")
    if args.artifact_sample_count is not None and args.artifact_sample_count < 1:
        parser.error("--artifact-sample-count must be at least 1")
    if args.nearest_neighbor_count < 1:
        parser.error("--nearest-neighbor-count must be at least 1")
    if args.nearest_neighbor_reference_limit < 1:
        parser.error("--nearest-neighbor-reference-limit must be at least 1")
    if args.lpips_pair_count < 1:
        parser.error("--lpips-pair-count must be at least 1")
    if args.cfg_comparison_scales is not None and any(scale < 0.0 for scale in args.cfg_comparison_scales):
        parser.error("--cfg-comparison-scales values must all be non-negative")
    return args


def main() -> None:
    args = parse_args()
    evaluation_config = CheckpointEvaluationConfig(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        output_dir=args.output_dir,
        run_name=args.run_name,
        data_dir=args.data_dir,
        num_generated_samples=args.num_generated_samples,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        sampler=args.sampler,
        sampling_steps=args.sampling_steps,
        ddim_eta=args.ddim_eta,
        guidance_scale=args.guidance_scale,
        amp_dtype=args.amp_dtype,
        artifact_sample_count=args.artifact_sample_count,
        cfg_comparison_scales=tuple(args.cfg_comparison_scales) if args.cfg_comparison_scales is not None else None,
        save_raw_images=args.save_raw_images,
        paired_metrics=args.paired_metrics,
        reference_stats_dir=args.reference_stats_dir,
        allow_model_download=args.allow_model_download,
        nearest_neighbor_count=args.nearest_neighbor_count,
        nearest_neighbor_reference_limit=args.nearest_neighbor_reference_limit,
        force_unconditional=args.force_unconditional,
        lpips_pair_count=args.lpips_pair_count,
    )
    payload = run_checkpoint_evaluation(
        evaluation_config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print("Evaluation directory:")
    print(payload["evaluation_dir"])
    print("Metrics file:")
    print(payload["metrics_path"])


if __name__ == "__main__":
    main()
