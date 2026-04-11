# Image Reconstruction Training

This repository trains `ae`, `dae`, `vae`, and `diffusion` models on MNIST-family datasets. The training entrypoint is now safe for shared Slurm/HPC usage:

- dataset roots are configurable with `--data_dir`
- downloads are opt-in with `--download`
- every run gets an isolated output directory
- logs go to both stdout and `train.log`
- resolved config is saved for reproducibility

## CLI overview

The primary entrypoint is [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py).

Common options:

- `--model {ae,dae,vae,diffusion,all}`
- `--dataset {mnist,fashion,fashion-mnist,fashion_mnist}`
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
- `--base_channels`
- `--time_dim`
- `--beta_start`
- `--beta_end`
- `--sample_count`

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

If the dataset is missing and `--download` is not set, training fails with a clear message telling you to pre-download on a login node or re-run with `--download`.

## Local experiment

Quick local diffusion test:

```bash
python3 train.py \
  --model diffusion \
  --dataset mnist \
  --epochs 5 \
  --batch_size 128 \
  --timesteps 200 \
  --base_channels 16 \
  --sample_count 8
```

The old simple style still works:

```bash
python3 train.py --model diffusion --epochs 5
```

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
- `checkpoints/best.pt`: best model checkpoint
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
