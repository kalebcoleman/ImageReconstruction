# Autoencoders

This package contains the learned reconstruction models used for the main
project:

- `models.py`: fully connected AE and VAE modules.
- `data.py`: MNIST and Fashion-MNIST dataset loading for 28x28 grayscale
  autoencoder experiments.
- `training.py`: AE/DAE/VAE train loops, VAE loss, denoising noise injection,
  and reconstruction metrics.
- `artifacts.py`: reconstruction grids, latent-space plots, VAE samples, and
  VAE interpolation figures.

Use `train.py` as the CLI entry point:

```bash
python train.py --model ae --dataset mnist --latent-dim 16
python train.py --model dae --dataset mnist --latent-dim 16 --dae-noise-level 0.2
python train.py --model vae --dataset mnist --latent-dim 16
```
