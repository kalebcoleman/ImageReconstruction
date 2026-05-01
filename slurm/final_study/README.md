# Final Diffusion Slurm Scripts

These scripts run the extra-credit diffusion models directly through
`train.py`. The cleanup removed Fashion-MNIST diffusion, checkpoint-only
FID/LPIPS evaluation, cross-run comparison orchestration, and finalization jobs.

Default matrix:

- `mnist`, seeds `1 2 3`
- `cifar10`, seeds `1 2 3`

## Defaults

```bash
CONDA_ENV=diffusion
REPO_DIR=$HOME/projects/ImageReconstruction
DATA_DIR=/scratch/$USER/image-reconstruction/data
STUDY_DIR=/scratch/$USER/image-reconstruction-final-study
```

## Submit

```bash
sbatch slurm/final_study/smoke_array.slurm
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

Each training array has three tasks by default. Override seeds with:

```bash
SEEDS="1 2 3" sbatch slurm/final_study/train_mnist_array.slurm
```

## Outputs

Completed runs are written under:

```text
$STUDY_DIR/runs/<dataset>/diffusion/<run_name>/
```

Useful files:

```text
checkpoints/best.pt
metrics.json
plots/loss_curve.png
plots/reconstructions.png
plots/diffusion_snapshots.png
samples/generated_samples.png
```

The scripts skip a run if `checkpoints/best.pt` and `metrics.json` already
exist. If a task left an incomplete run directory, rerun with:

```bash
CLEAR_INCOMPLETE=1 sbatch --array=<task_id> slurm/final_study/train_mnist_array.slurm
CLEAR_INCOMPLETE=1 sbatch --array=<task_id> slurm/final_study/train_cifar10_array.slurm
```

## GPU Overrides

```bash
sbatch --constraint=h200 slurm/final_study/train_cifar10_array.slurm
sbatch --constraint=v100 slurm/final_study/train_mnist_array.slurm
sbatch --constraint=rtx6000 slurm/final_study/smoke_array.slurm
```

## Logs

Logs are written under:

```text
slurm/final_study/logs/
```

Useful checks:

```bash
ls -lt slurm/final_study/logs/
tail -n 80 slurm/final_study/logs/final_train_mnist_<job>_<task>.out
tail -n 80 slurm/final_study/logs/final_train_cifar10_<job>_<task>.out
```
