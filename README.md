# Image Reconstruction

This project compares PCA and autoencoder-based methods for image
reconstruction across MNIST, Fashion-MNIST, and CIFAR-10. PCA is the classical
baseline handled by my teammate; this side focuses on regular autoencoders,
denoising autoencoders, VAEs, latent interpolation/generation experiments, and
a small diffusion extension for extra credit.

The final project writeup lives at [`docs/index.md`](docs/index.md).

## Code Layout

- `autoencoders/`: AE, DAE, VAE models, train/eval loops, metrics, and
  autoencoder-specific plots.
- `diffusion/`: DDPM models, scheduler, sampler, diffusion train/eval loops,
  and diffusion-specific image artifacts.
- `train.py`: shared CLI that selects either the autoencoder package or the
  diffusion package.
- `configs/diffusion/`: MNIST and CIFAR10 diffusion recipes.

AE/DAE/VAE are the main learned reconstruction work. Diffusion is extra credit
only, kept to MNIST and CIFAR10 after cleanup. Fashion-MNIST is still supported
by the autoencoder CLI, but it is no longer part of the diffusion workflow.

## Autoencoder Commands

Regular autoencoder:

```bash
python train.py --model ae --dataset mnist --latent-dim 16
```

Denoising autoencoder:

```bash
python train.py --model dae --dataset mnist --latent-dim 16 --dae-noise-level 0.2
```

Variational autoencoder:

```bash
python train.py --model vae --dataset mnist --latent-dim 16
```

Run all autoencoder variants on MNIST:

```bash
python train.py --models ae dae vae --dataset mnist --latent-dim 16
```

Fashion-MNIST still works for the autoencoder side:

```bash
python train.py --model ae --dataset fashion --latent-dim 16
```

## Diffusion Commands

The cleaned diffusion path is intentionally small:

- `mnist`: legacy diffusion, native `28x28`, `1` channel.
- `cifar10`: ADM diffusion, native `32x32`, `3` channels.

The old checkpoint-only FID/LPIPS/Inception evaluation path was removed. The
diffusion deliverable is now simply: train the model, save the checkpoint,
loss curve, reconstructions, reverse-process snapshots, and generated samples.

Manual diffusion runs:

```bash
python train.py --config configs/diffusion/mnist.yaml --data-dir /scratch/$USER/image-reconstruction/data
python train.py --config configs/diffusion/cifar10.yaml --data-dir /scratch/$USER/image-reconstruction/data
```

## Repository Structure

```text
.
├── diffusion/              # DDPM model, scheduler, training, and sampling code
├── autoencoders/           # AE/DAE/VAE model, training, metrics, and artifacts
├── configs/diffusion/      # MNIST and CIFAR10 diffusion recipes
├── slurm/final_study/      # Simple Slurm train arrays
├── scripts/                # Utility scripts for reports and visualizations
├── docs/                   # GitHub Pages project website
├── train.py                # Main training entry point
└── README.md
```

## Diffusion Slurm Commands

Smoke check:

```bash
sbatch slurm/final_study/smoke_array.slurm
```

Final diffusion training:

```bash
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

The scripts write outputs under:

```text
/scratch/$USER/image-reconstruction-final-study/runs/<dataset>/diffusion/<run_name>/
```

Each completed run should contain:

```text
checkpoints/best.pt
metrics.json
plots/loss_curve.png
plots/reconstructions.png
plots/diffusion_snapshots.png
samples/generated_samples.png
```

Rerun a failed or incomplete task with:

```bash
CLEAR_INCOMPLETE=1 sbatch --array=<task_id> slurm/final_study/train_mnist_array.slurm
CLEAR_INCOMPLETE=1 sbatch --array=<task_id> slurm/final_study/train_cifar10_array.slurm
```

Use `sbatch --constraint=h200`, `--constraint=v100`, or
`--constraint=rtx6000` to switch GPU types.

## Report Assets

Report assets for GitHub Pages can be copied into `docs/assets/` with:

```bash
python scripts/collect_report_assets.py
```

Diffusion should be reported separately from the PCA vs AE/DAE/VAE comparison.
The main project remains reconstruction; diffusion is extra credit generation.
