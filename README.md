# Image Reconstruction Training

This repository trains `ae`, `dae`, `vae`, and `diffusion` models through a single [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py) entrypoint. Phase 1 of the diffusion refactor keeps the legacy autoencoder paths intact while expanding diffusion to MNIST, FashionMNIST, CIFAR10, and ImageNet-ready dataset adapters with a scalable ADM-style U-Net backend.

Phase 2 extends the new ADM diffusion path with classifier-free guidance, configurable attention resolutions, `eps` / `v` prediction targets, DDPM + DDIM sampling, and mixed-precision / gradient-clipping controls for more realistic research-style runs on shared HPC systems.

The training entrypoint remains safe for shared Slurm/HPC usage:

- dataset roots are configurable with `--data_dir`
- downloads are opt-in with `--download`
- every run gets an isolated output directory
- logs go to both stdout and `train.log`
- resolved config is saved for reproducibility

## CLI overview

The primary entrypoint is [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py).

Common options:

- `--config /path/to/recipe.yaml`
- `--model {ae,dae,vae,diffusion,all}`
- `--dataset {mnist,fashion,fashion-mnist,fashion_mnist,cifar10,cifar,cifar-10,cifar_10,imagenet,ilsvrc,ilsvrc2012}`
- `--epochs`
- `--batch_size`
- `--lr`
- `--seed`
- `--data_dir`
- `--output_dir`
- `--run_name`
- `--num_workers`
- `--download`

Model-specific options:

- `--latent_dim` for `ae`, `dae`, and `vae`
- `--timesteps`
- `--diffusion_backbone {adm,legacy}` or `--legacy-diffusion`
- `--diffusion-preprocessing {default,parity_64}`
- `--image_size`
- `--diffusion_channels`
- `--base_channels`
- `--time_dim`
- `--schedule {linear,cosine}`
- `--beta_start`
- `--beta_end`
- `--ema_decay`
- `--num_res_blocks`
- `--prediction_type {eps,v}`
- `--attention_resolutions 16 8`
- `--class_dropout_prob`
- `--guidance_scale`
- `--sampler {ddpm,ddim}`
- `--sampling_steps`
- `--ddim_eta`
- `--grad_clip_norm`
- `--amp_dtype {auto,none,bf16,fp16}`
- `--sample_count`

Notes:

- `--timesteps` is the diffusion process length and still supports settings such as `500` and `1000`.
- `--time_dim` is only the timestep embedding width inside the UNet.
- `--beta_start` and `--beta_end` shape the linear schedule; the cosine schedule uses the standard improved-DDPM cosine curve.
- The new diffusion default is `--diffusion_backbone adm`, which resolves to `64x64` and `3` channels unless you override them.
- `--config` loads a YAML recipe first, then applies any explicit CLI flags on top of it.
- The old diffusion path is still available through `--legacy-diffusion`; on MNIST/Fashion it resolves to native `28x28` grayscale by default.
- `--prediction_type eps` preserves the earlier behavior. Use `--prediction_type v` for the stronger phase-2 objective.
- `--sampler ddpm` preserves the earlier ancestral sampler behavior. `--sampler ddim --sampling_steps 50` is the main new fast-sampling path.
- `--amp_dtype auto` prefers `bf16` on supported CUDA GPUs and falls back to `fp16` otherwise.
- `ae`, `dae`, and `vae` remain MNIST/Fashion-only paths for now. CIFAR10 and ImageNet are diffusion-only in this phase.

## Parity Protocol

Phase 4 adds a locked recipe family under [`configs/diffusion/`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/). The parity protocol is intended for fair cross-dataset comparison of the same pixel-space ADM diffusion family.

Identical across the parity recipes:

- `image_size = 64`
- `diffusion_channels = 3`
- `diffusion_backbone = adm`
- `diffusion_preprocessing = parity_64`
- `prediction_type = v`
- `schedule = cosine`
- `ema_decay = 0.999`
- `class_dropout_prob = 0.1`
- `sampler = ddim`
- `sampling_steps = 50`
- `ddim_eta = 0.0`
- `attention_resolutions = [16, 8]`

Allowed to differ across datasets:

- `batch_size`
- `num_workers`
- `epochs`
- `eval_batch_size`
- `data_dir`
- `output_dir`
- `run_name`
- the concrete ImageNet subset/full-data realization behind `data_dir`

The parity recipes are:

- [`configs/diffusion/base_adm64.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/base_adm64.yaml)
- [`configs/diffusion/mnist_64.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/mnist_64.yaml)
- [`configs/diffusion/fashion_64.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/fashion_64.yaml)
- [`configs/diffusion/cifar10_64.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/cifar10_64.yaml)
- [`configs/diffusion/imagenet_64.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/imagenet_64.yaml)

The lightweight smoke-test recipes live under [`configs/diffusion/smoke/`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/). They keep the same ADM/parity preprocessing path and output schema, but reduce runtime with `1` epoch, `100` diffusion timesteps, fewer generated samples, smaller artifact grids, and smaller CFG comparison defaults.

The ImageNet recipe is a protocol definition for `64x64` parity runs. It does not claim that the repo has already validated a full ImageNet benchmark result in this phase.

## Final Study Runner

Phase 5 adds [`run_parity_suite.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/run_parity_suite.py), a lightweight orchestration layer for the final seeded study on:

- `mnist`
- `fashion`
- `cifar10`

Default final-study behavior:

- `3` seeds per dataset
- deterministic run names like `parity_mnist_64_seed001`
- train/eval linkage recorded in `study_registry.json`
- non-destructive safety checks for existing run directories
- optional `--skip-existing` reuse for completed train/eval steps
- summary regeneration after execution by default

The final-study runner does not replace [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py) or [`evaluate.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/evaluate.py). It calls them with deterministic config/seed/run-name combinations.

Smoke-test behavior with `--smoke`:

- uses the lightweight recipes under [`configs/diffusion/smoke/`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/)
- defaults to seed `1` when `--seeds` is omitted
- uses distinct run names such as `parity_mnist_64_smoke_seed001`
- keeps the same `runs/`, `summaries/`, registry, selection, and deliverables layout
- refuses to mix smoke and full-study runs in the same `--study-dir`
- if pretrained evaluation weights are not cached yet, rerun the eval-inclusive command with `--allow-model-download` once

Plan shell and Slurm-friendly commands without running:

```bash
python3 run_parity_suite.py plan \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase both
```

Quick smoke validation for MNIST:

```bash
python3 run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-final-study-smoke-mnist \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets mnist \
  --phase both

python3 run_parity_suite.py deliverables \
  --study-dir /scratch/$USER/image-reconstruction-final-study-smoke-mnist
```

Quick smoke validation for FashionMNIST:

```bash
python3 run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-final-study-smoke-fashion \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets fashion \
  --phase both

python3 run_parity_suite.py deliverables \
  --study-dir /scratch/$USER/image-reconstruction-final-study-smoke-fashion
```

Quick smoke validation for CIFAR10:

```bash
python3 run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-final-study-smoke-cifar10 \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets cifar10 \
  --phase both

python3 run_parity_suite.py deliverables \
  --study-dir /scratch/$USER/image-reconstruction-final-study-smoke-cifar10
```

Run the unchanged full final study:

```bash
python3 run_parity_suite.py run \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase both
```

If the first evaluation on a machine needs to cache Inception/LPIPS weights, add `--allow-model-download` to the `run` or `plan` command once:

```bash
python3 run_parity_suite.py run \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase both \
  --allow-model-download
```

Run train-only:

```bash
python3 run_parity_suite.py run \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase train
```

Run eval-only:

```bash
python3 run_parity_suite.py run \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase eval
```

Reuse completed train/eval outputs instead of erroring:

```bash
python3 run_parity_suite.py run \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase both \
  --skip-existing
```

Regenerate summaries:

```bash
python3 run_parity_suite.py summarize \
  --study-dir /scratch/$USER/image-reconstruction-final-study
```

Select best and median runs per dataset by FID:

```bash
python3 run_parity_suite.py select-best \
  --study-dir /scratch/$USER/image-reconstruction-final-study
```

Outputs added by the study runner:

- `study_registry.json` plus YAML/Markdown views
- per-run study summary
- per-dataset mean/std summary across seeds
- best/median run selection tables
- final study Markdown report
- planned command files for shell and Slurm array submission

## Final Deliverables

Phase 6 adds a polished deliverables export layer for the completed final study. It gathers:

- main results table
- mean/std table across seeds
- best-run table per dataset
- artifact index
- best exported figures with stable names
- report-ready markdown summary
- presentation figure index

Generate the full final deliverables bundle:

```bash
python3 run_parity_suite.py deliverables \
  --study-dir /scratch/$USER/image-reconstruction-final-study
```

Regenerate the final markdown summaries only:

```bash
python3 run_parity_suite.py summarize \
  --study-dir /scratch/$USER/image-reconstruction-final-study
```

Export the best artifacts per dataset into one stable folder:

```bash
python3 run_parity_suite.py export-best-artifacts \
  --study-dir /scratch/$USER/image-reconstruction-final-study
```

Default deliverables bundle layout:

```text
{study_dir}/deliverables/
├── deliverables_bundle.json
├── deliverables_bundle.yaml
├── deliverables_bundle.md
├── figures/
│   ├── mnist_best_generated_samples.png
│   ├── mnist_cfg_comparison.png
│   ├── mnist_reverse_process_snapshots.png
│   ├── mnist_nearest_neighbors.png
│   └── ...
├── tables/
│   ├── main_results_table.csv
│   ├── mean_std_table.csv
│   ├── best_runs_table.csv
│   └── artifact_index.csv
└── summaries/
    ├── analysis_summary.md
    ├── project_report_summary.md
    └── presentation_figure_index.md
```

Metric interpretation for the final deliverables:

- `FID` is the primary generative metric.
- `Inception Score` is secondary.
- `LPIPS diversity` is a perceptual diversity signal, not a fidelity metric.
- `PSNR` and `SSIM` remain auxiliary paired denoising metrics only.

Best-run selection:

- best run per dataset = minimum `FID`
- median run per dataset = sort by `FID`, choose index `floor(n/2)`

Aggregation:

- cross-run evaluation payloads are collected by [`aggregate_results.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/aggregate_results.py)
- study-level summaries and final bundle exports are layered on top of those same evaluation payloads, not reconstructed from logs

## Pre-download datasets

On a login node, pre-download the datasets once into a shared path. The `--download` flag is intentionally off by default for compute-node safety.

MNIST:

```bash
python3 train.py \
  --model ae \
  --dataset mnist \
  --epochs 1 \
  --batch_size 512 \
  --data_dir /shared/datasets/image-reconstruction \
  --output_dir /tmp/image-reconstruction-bootstrap \
  --download
```

Fashion-MNIST:

```bash
python3 train.py \
  --model ae \
  --dataset fashion \
  --epochs 1 \
  --batch_size 512 \
  --data_dir /shared/datasets/image-reconstruction \
  --output_dir /tmp/image-reconstruction-bootstrap \
  --download
```

CIFAR-10:

```bash
python3 train.py \
  --model diffusion \
  --dataset cifar10 \
  --epochs 1 \
  --batch_size 256 \
  --data_dir /shared/datasets/image-reconstruction \
  --output_dir /tmp/image-reconstruction-bootstrap \
  --download
```

ImageNet setup is manual in this phase. Prepare:

```text
{data_dir}/imagenet/train/<class_name>/*.JPEG
{data_dir}/imagenet/val/<class_name>/*.JPEG
```

`--download` is intentionally unsupported for ImageNet, and the trainer will fail with a clear path hint if those directories are missing.

If the dataset is missing and `--download` is not set, training fails with a clear message telling you to pre-download on a login node or re-run with `--download`.

## Local experiment

Quick local ADM diffusion tests:

```bash
python3 train.py \
  --model diffusion \
  --dataset mnist \
  --epochs 5 \
  --batch_size 64 \
  --timesteps 1000 \
  --base_channels 64 \
  --time_dim 128 \
  --schedule cosine \
  --ema_decay 0.999 \
  --num_res_blocks 2 \
  --prediction_type v \
  --attention-resolutions 16 8 \
  --class_dropout_prob 0.1 \
  --guidance_scale 3.0 \
  --sampler ddim \
  --sampling_steps 50 \
  --grad_clip_norm 1.0 \
  --amp_dtype auto \
  --sample_count 8
```

FashionMNIST:

```bash
python3 train.py \
  --model diffusion \
  --dataset fashion \
  --epochs 5 \
  --batch_size 64 \
  --timesteps 1000 \
  --image_size 64 \
  --diffusion_channels 3 \
  --base_channels 64 \
  --time_dim 128 \
  --schedule cosine \
  --ema_decay 0.999 \
  --num_res_blocks 2 \
  --prediction_type v \
  --attention-resolutions 16 8 \
  --class_dropout_prob 0.1 \
  --guidance_scale 3.0 \
  --sampler ddim \
  --sampling_steps 50 \
  --grad_clip_norm 1.0 \
  --amp_dtype auto \
  --sample_count 8
```

CIFAR-10:

```bash
python3 train.py \
  --model diffusion \
  --dataset cifar10 \
  --epochs 5 \
  --batch_size 64 \
  --timesteps 1000 \
  --image_size 64 \
  --diffusion_channels 3 \
  --base_channels 64 \
  --time_dim 128 \
  --schedule cosine \
  --ema_decay 0.999 \
  --num_res_blocks 2 \
  --prediction_type v \
  --attention-resolutions 16 8 \
  --class_dropout_prob 0.1 \
  --guidance_scale 3.0 \
  --sampler ddim \
  --sampling_steps 50 \
  --grad_clip_norm 1.0 \
  --amp_dtype auto \
  --sample_count 8
```

The old simple style still works, now on the ADM default:

```bash
python3 train.py --model diffusion --dataset mnist --epochs 5
```

Legacy compatibility run:

```bash
python3 train.py \
  --model diffusion \
  --dataset mnist \
  --legacy-diffusion \
  --image_size 28 \
  --diffusion_channels 1 \
  --epochs 5
```

Standardized parity runs:

MNIST:

```bash
python3 train.py --config configs/diffusion/mnist_64.yaml \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --output-dir /scratch/$USER/image-reconstruction-runs/parity
```

FashionMNIST:

```bash
python3 train.py --config configs/diffusion/fashion_64.yaml \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --output-dir /scratch/$USER/image-reconstruction-runs/parity
```

CIFAR10:

```bash
python3 train.py --config configs/diffusion/cifar10_64.yaml \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --output-dir /scratch/$USER/image-reconstruction-runs/parity
```

ImageNet parity protocol:

```bash
python3 train.py --config configs/diffusion/imagenet_64.yaml \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --output-dir /scratch/$USER/image-reconstruction-runs/parity
```

## Checkpoint Evaluation

Phase 3 adds a separate [`evaluate.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/evaluate.py) entrypoint for checkpoint-only evaluation and sampling. The training path in [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py) is unchanged.

Primary generative metrics:

- `FID` as the primary reported score
- `Inception Score` as a secondary generative score
- `LPIPS diversity` as a perceptual diversity signal across generated-image pairs

Auxiliary paired metrics:

- `PSNR`
- `SSIM`

Those paired metrics are still useful for denoising-style sanity checks, but they are not reported as the primary generative comparison.

Evaluation command pattern:

```bash
python3 evaluate.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --mode evaluate \
  --num-generated-samples 1000 \
  --eval-batch-size 64 \
  --sampler ddim \
  --sampling-steps 50 \
  --ddim-eta 0.0 \
  --guidance-scale 3.0 \
  --artifact-sample-count 16 \
  --cfg-comparison-scales 0 1 3 5 \
  --amp-dtype auto
```

Sampling-only command pattern:

```bash
python3 evaluate.py \
  --checkpoint /path/to/checkpoints/best.pt \
  --mode sample \
  --sampler ddim \
  --sampling-steps 50 \
  --guidance-scale 3.0 \
  --artifact-sample-count 16 \
  --save-raw-images
```

Notes:

- `evaluate.py` defaults to saving results under `<training_run>/evaluations/`.
- Real-image FID reference stats are cached under `<output_dir>/_reference_stats/` unless you override `--reference-stats-dir`.
- The same dataset/image-size/channel preprocessing signature is used for both cached real stats and generated images.
- When a checkpoint was trained from a parity recipe, `evaluate.py` can reuse the checkpoint’s saved evaluation defaults such as `eval_batch_size`, `num_generated_samples`, and CFG comparison scales.
- If the required torchvision model weights are not already cached locally, rerun on a login node with `--allow-model-download` once, then use the cached weights on compute nodes.

## Aggregation

Phase 4 adds [`aggregate_results.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/aggregate_results.py) to combine completed evaluation payloads into one comparison table.

Example:

```bash
python3 aggregate_results.py \
  /scratch/$USER/image-reconstruction-runs/parity \
  --output-dir /scratch/$USER/image-reconstruction-runs/parity_reports/final_table
```

This writes:

- `aggregated_results.json`
- `aggregated_results.yaml`
- `aggregated_results.csv`
- `aggregated_results.md`
- `comparison_table.md`

## Slurm experiment

Single Slurm-safe run:

```bash
python3 train.py \
  --model diffusion \
  --dataset mnist \
  --epochs 25 \
  --batch_size 256 \
  --lr 1e-3 \
  --seed 42 \
  --data_dir /scratch/$USER/image-reconstruction/data \
  --output_dir /scratch/$USER/image-reconstruction-runs \
  --run_name slurm_job_${SLURM_JOB_ID} \
  --num_workers 4 \
  --timesteps 200 \
  --base_channels 16 \
  --schedule cosine \
  --ema_decay 0.999 \
  --num_res_blocks 2 \
  --beta_start 1e-4 \
  --beta_end 2e-2 \
  --sample_count 16
```

For the curated HPC-friendly scripts, see [`slurm/README.md`](/Users/itzjuztmya/Kaleb/ImageReconstruction/slurm/README.md). The recommended order is:

1. `sbatch slurm/test_diffusion.slurm`
2. `sbatch slurm/array_diffusion_medium.slurm`
3. `sbatch slurm/array_diffusion_large.slurm` only if the medium sweep behaves well and you want the heavier 1000-timestep study

Example `sbatch` snippet:

```bash
#!/bin/bash
#SBATCH --job-name=mnist-diffusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

module load python
source .venv/bin/activate

python3 train.py \
  --model diffusion \
  --dataset mnist \
  --epochs 25 \
  --batch_size 256 \
  --lr 1e-3 \
  --seed 42 \
  --data_dir /scratch/$USER/image-reconstruction/data \
  --output_dir /scratch/$USER/image-reconstruction-runs \
  --run_name job_${SLURM_JOB_ID} \
  --num_workers ${SLURM_CPUS_PER_TASK:-4} \
  --timesteps 200 \
  --base_channels 16 \
  --schedule cosine \
  --ema_decay 0.999 \
  --num_res_blocks 2 \
  --sample_count 16
```

## Parameter sweeps

Four-run sweep driven from the shell:

```bash
for seed in 1 2; do
  for channels in 16 32; do
    python3 train.py \
      --model diffusion \
      --dataset mnist \
      --epochs 25 \
      --batch_size 256 \
      --lr 1e-3 \
      --seed "${seed}" \
      --data_dir /scratch/$USER/image-reconstruction/data \
      --output_dir /scratch/$USER/image-reconstruction-runs \
      --run_name seed${seed}_ch${channels} \
      --num_workers 4 \
      --timesteps 200 \
      --base_channels "${channels}" \
      --schedule cosine \
      --ema_decay 0.999 \
      --num_res_blocks 2 \
      --sample_count 16
  done
done
```

You can also preserve the old multi-model behavior with:

```bash
python3 train.py --model all --dataset mnist
```

That runs each model as a separate isolated run directory.

## Output structure

Each run resolves to:

```text
{output_dir}/{dataset}/{model}/{run_name_or_auto_name}/
```

Example:

```text
/scratch/alice/image-reconstruction-runs/mnist/diffusion/mnist_diffusion_t200_ch16_bs256_lr1e-3_seed42_2026-04-11_153000_123456/
```

Each run directory contains:

- `config.json`: resolved config and CLI args
- `train.log`: batch-friendly file log
- `metrics.jsonl`: epoch-by-epoch metrics
- `metrics.json`: final summary
- `kfold_results.csv`: compact summary CSV
- `checkpoints/best.pt`: best model checkpoint, plus `ema_state_dict` when EMA is enabled
- `plots/`: saved visuals such as `loss_curve.png`, diffusion `reconstructions.png`, and `diffusion_snapshots.png`
- `samples/`: generated sample grids such as diffusion `generated_samples.png`

For diffusion runs, successful jobs now always save visible artifacts in the run directory and also mirror the main images into the familiar legacy layout:

- `plots/loss_curve.png`
- `plots/reconstructions.png`
- `plots/diffusion_snapshots.png`
- `samples/generated_samples.png`
- `outputs/diffusion/loss_curves/<run_name>.png`
- `outputs/diffusion/reconstructions/<run_name>.png`
- `outputs/diffusion/snapshots/<run_name>.png`
- `outputs/diffusion/samples/<run_name>.png`

That guarantee applies even to short smoke tests, so a 1-epoch Slurm validation run still leaves behind report/demo-friendly images.

If a target run directory already exists, the trainer appends a numeric suffix instead of overwriting it.

## Notes

- Use `--output_dir` to point Slurm jobs at scratch or project storage instead of the repo checkout.
- `plot_kfold.py` now accepts `--results_csv` and `--output_path` so plotting can also target a specific run directory.
