# Final Study Slurm Arrays

These scripts run the final dataset-appropriate diffusion study on Monsoon with one Slurm task per independent dataset/seed unit.

The default final matrix is:

- `mnist`, seeds `1 2 3`
- `fashion`, seeds `1 2 3`
- `cifar10`, seeds `1 2 3`

ImageNet is intentionally not in the default final study.

## Why Dataset-Specific Training Arrays

Slurm arrays share one wall-time limit across all tasks. MNIST and FashionMNIST are expected to fit in shorter allocations than CIFAR10, so final training is split into dataset-specific arrays:

- `train_mnist_array.slurm`: A100, 8h, 4 CPUs, 32G
- `train_fashion_array.slurm`: A100, 8h, 4 CPUs, 32G
- `train_cifar10_array.slurm`: A100, 24h, 4 CPUs, 32G

This keeps CIFAR10 from forcing MNIST and FashionMNIST to reserve 24 hours.

## Resource Defaults

- Smoke: A100, 2h, 4 CPUs, 32G
- MNIST/Fashion training: A100, 8h, 4 CPUs, 32G
- CIFAR10 training: A100, 24h, 4 CPUs, 32G
- Evaluation: A100, 12h, 4 CPUs, 32G

Every script defaults to:

```bash
CONDA_ENV=diffusion
REPO_DIR=$HOME/projects/ImageReconstruction
DATA_DIR=/scratch/$USER/image-reconstruction/data
STUDY_DIR=/scratch/$USER/image-reconstruction-final-study
GPU_CONSTRAINT=a100
PARTITION=gpu
```

The smoke script uses `STUDY_DIR=/scratch/$USER/image-reconstruction-final-study-smoke-all`.

## Submit Smoke

```bash
sbatch slurm/final_study/smoke_array.slurm
```

The smoke array runs train-only, one task each for `mnist`, `fashion`, and `cifar10`, with `--smoke --phase train`. Smoke outputs use 1 epoch and are expected to be blurry or noisy; they are only pipeline checks for environment, data paths, configs, logging, and output writing.

## Submit Final Training

```bash
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_fashion_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

Each training array has three tasks by default, one for each seed `1`, `2`, and `3`. The scripts pass `--skip-existing` and `--no-summarize` so completed deterministic runs are reused and array tasks do not compete to write summaries.

## Submit Evaluation

First evaluation, if metric weights are not cached:

```bash
ALLOW_MODEL_DOWNLOAD=1 sbatch slurm/final_study/eval_all_array.slurm
```

Normal evaluation after metric weights are cached:

```bash
sbatch slurm/final_study/eval_all_array.slurm
```

The eval array has nine tasks by default: `3 datasets x 3 seeds`. `ALLOW_MODEL_DOWNLOAD=1` is only passed through as `--allow-model-download` when requested.

## Finalize

```bash
sbatch slurm/final_study/finalize.slurm
```

The finalize job runs:

```bash
python run_parity_suite.py summarize --study-dir "$STUDY_DIR"
python run_parity_suite.py select-best --study-dir "$STUDY_DIR"
python run_parity_suite.py export-best-artifacts --study-dir "$STUDY_DIR"
python run_parity_suite.py deliverables --study-dir "$STUDY_DIR"
```

## Logs

Logs are written under:

```bash
slurm/final_study/logs/
```

Useful checks:

```bash
ls -lt slurm/final_study/logs/
tail -n 80 slurm/final_study/logs/final_train_mnist_<job>_<task>.out
tail -n 80 slurm/final_study/logs/final_eval_all_<job>_<task>.err
```

## Rerun Failed Tasks

The array scripts already pass `--skip-existing`, so rerunning an array will skip completed deterministic outputs and run missing ones:

```bash
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/eval_all_array.slurm
```

To rerun only one failed task, use `--array`:

```bash
sbatch --array=1 slurm/final_study/train_fashion_array.slurm
sbatch --array=8 slurm/final_study/eval_all_array.slurm
```

If a failed task left an incomplete deterministic output directory, clear only that incomplete directory while preserving completed runs:

```bash
CLEAR_INCOMPLETE=1 sbatch --array=1 slurm/final_study/train_fashion_array.slurm
CLEAR_INCOMPLETE=1 sbatch --array=8 slurm/final_study/eval_all_array.slurm
```

## GPU Constraint Overrides

The scripts include default `#SBATCH --partition=gpu` and `#SBATCH --constraint=a100` directives. Slurm reads those before the shell starts, so resource changes must be made with `sbatch` options:

```bash
sbatch --constraint=h200 slurm/final_study/train_cifar10_array.slurm
sbatch --constraint=v100 slurm/final_study/train_mnist_array.slurm
sbatch --constraint=rtx6000 slurm/final_study/smoke_array.slurm
```

For a generic GPU run, override or remove the constraint at submission time:

```bash
sbatch --constraint="" slurm/final_study/smoke_array.slurm
```

If your Slurm version rejects an empty constraint, make a local copy of the script and remove the `#SBATCH --constraint=a100` line for that submission. Use `--partition=<name>` the same way if Monsoon partition names differ.

## Dry Run

Each script supports `DRY_RUN=1` for command and task-mapping validation without activating conda or launching Python:

```bash
DRY_RUN=1 SLURM_ARRAY_TASK_ID=8 bash slurm/final_study/eval_all_array.slurm
```
