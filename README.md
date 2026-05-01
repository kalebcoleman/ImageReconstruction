# Image Reconstruction

CS 472 unsupervised learning final project comparing PCA and autoencoder-based image reconstruction on MNIST and Fashion-MNIST. PCA is teammate-owned work and final PCA figures/analysis will be merged later. This side of the repo contains AE, DAE, VAE, VAE interpolation/generation, and an extra-credit diffusion extension.

The GitHub Pages writeup is in [`docs/index.md`](docs/index.md).

## Project Layout

```text
autoencoders/              AE, DAE, VAE models, training, metrics, plots
diffusion/                 Extra-credit DDPM code for MNIST and CIFAR-10
configs/diffusion/         mnist.yaml and cifar10.yaml diffusion configs
scripts/collect_report_assets.py
docs/                      GitHub Pages site and small final image assets
slurm/final_study/         Reproducible diffusion smoke/MNIST/CIFAR jobs
train.py                   Main training entry point
requirements.txt           Python dependencies
```

Generated folders such as `data/`, `outputs/`, `runs/`, `checkpoints/`, and `deliverables/` are ignored.

## Autoencoder Runs

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

Run all three learned reconstruction models:

```bash
python train.py --models ae dae vae --dataset mnist --latent-dim 16
```

Fashion-MNIST is supported for the autoencoder side:

```bash
python train.py --model ae --dataset fashion --latent-dim 16
```

Autoencoder outputs are written under:

```text
outputs/<dataset>/<model>/<run_name>/
```

## Diffusion Extra Credit

Diffusion is an extra-credit extension, not the main PCA vs autoencoder comparison. The active diffusion workflow is intentionally limited to:

- MNIST: native `28x28`, grayscale, legacy U-Net config
- CIFAR-10: native `32x32`, RGB, ADM-style U-Net config

Fashion-MNIST diffusion and old checkpoint-only FID/LPIPS/Inception evaluation paths are not part of the cleaned workflow. A diffusion run trains the model and saves checkpoints, metrics, a loss curve, reconstructions, reverse-process snapshots, native sample grids, and generated samples.

Run from configs:

```bash
python train.py --config configs/diffusion/mnist.yaml
python train.py --config configs/diffusion/cifar10.yaml
```

Diffusion outputs are written under:

```text
outputs/<dataset>/diffusion/<run_name>/
```

Expected files include:

```text
checkpoints/best.pt
metrics.json
plots/loss_curve.png
plots/reconstructions.png
plots/diffusion_snapshots.png
samples/generated_samples.png
samples/generated_samples_native_grid.png
```

## Slurm Diffusion Runs

The small final-study Slurm surface is under `slurm/final_study/`:

```bash
sbatch slurm/final_study/smoke_array.slurm
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

By default these write to:

```text
/scratch/$USER/image-reconstruction-final-study/runs/<dataset>/diffusion/<run_name>/
```

## Report Assets

Copy small, final report images into `docs/assets/`:

```bash
python scripts/collect_report_assets.py
```

The script creates PCA placeholders, collects AE/DAE/VAE figures when available, and prepares native plus nearest-neighbor-scaled MNIST/CIFAR diffusion sample grids for GitHub Pages.

## PCA Status

PCA results are intentionally placeholders for now. The final report still needs teammate PCA reconstruction grids, PCA metrics, and the final PCA vs AE conclusion section.
