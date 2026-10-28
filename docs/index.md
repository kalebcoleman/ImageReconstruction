---
---
# PCA vs Autoencoders for Image Reconstruction

<link rel="stylesheet" href="./style.css">

## Overview

This CS 472 final project compares classical PCA reconstruction with neural autoencoder reconstruction on MNIST and Fashion-MNIST. PCA assets and the final cross-method analysis are owned by my teammate and have not yet been added. This page presents the available autoencoder results and the diffusion extension.

My portion covers regular autoencoders, denoising autoencoders, variational autoencoders, VAE interpolation/generation, and a small diffusion extension. Diffusion is extra credit and is separate from the main PCA vs autoencoder comparison.

## Main Comparison: PCA vs Autoencoder

The intended comparison uses matched compression levels, visual grids, and metrics such as MSE, PSNR, and SSIM. PCA results are not yet available in this repository, so no cross-method conclusions are shown here.

### Current Autoencoder Reference Results

![MNIST AE reconstruction grid](./assets/mnist/ae_reconstruction_latent16.png)

![MNIST AE model comparison at latent dimension 16](./assets/mnist/model_comparison_latent16.png)

## AE / DAE / VAE Results

The regular autoencoder is the main neural reconstruction baseline against PCA. The denoising autoencoder tests robustness to noisy inputs, and the VAE adds a probabilistic latent space for sampling and interpolation.

### Regular Autoencoder

![MNIST AE reconstruction latent 16](./assets/mnist/ae_reconstruction_latent16.png)

![MNIST AE latent space latent 16](./assets/mnist/ae_latent_space_latent16.png)

![MNIST AE loss curve latent 16](./assets/mnist/ae_loss_latent16.png)

### Denoising Autoencoder

![MNIST DAE noisy reconstruction latent 16](./assets/mnist/dae_reconstruction_noise02_latent16.png)

![MNIST DAE loss curve latent 16](./assets/mnist/dae_loss_noise02_latent16.png)

### Variational Autoencoder

![MNIST VAE reconstruction latent 16](./assets/mnist/vae_reconstruction_latent16.png)

![MNIST VAE latent space latent 16](./assets/mnist/vae_latent_space_latent16.png)

![MNIST VAE generated samples](./assets/mnist/vae_generated_latent16.png)

## VAE Interpolation and Generation

The VAE supports sampling from its learned latent distribution, as shown in the generated samples above. An interpolation figure is not available in the repository yet.

## Metrics Snapshot

The model comparison above shows MSE, PSNR, and SSIM for the available MNIST autoencoder experiments at latent dimension 16. PCA metrics will be added when the teammate’s results are available.

## Extra Credit Extension: Diffusion Model

Diffusion is an extra-credit generative extension, not the main reconstruction comparison. The cleaned workflow keeps diffusion to MNIST and CIFAR-10 only. Fashion-MNIST diffusion runs were removed from the active workflow.

The native grids preserve the original generated sample size. The enlarged grids are pre-scaled with nearest-neighbor interpolation so CIFAR-10 does not look blurred by browser or plotting interpolation.

### MNIST Diffusion

Native 28x28 samples:

<img src="./assets/diffusion/mnist_samples_native.png" class="pixelated native-grid" width="280" height="280" alt="MNIST diffusion samples at native 28x28 grid size">

Nearest-neighbor 4x display:

<img src="./assets/diffusion/mnist_samples_nearest_4x.png" class="pixelated" alt="MNIST diffusion samples enlarged with nearest-neighbor interpolation">

### CIFAR-10 Diffusion

Native 32x32 samples:

<img src="./assets/diffusion/cifar10_samples_native.png" class="pixelated native-grid" width="320" height="320" alt="CIFAR-10 diffusion samples at native 32x32 grid size">

Nearest-neighbor 4x display:

<img src="./assets/diffusion/cifar10_samples_nearest_4x.png" class="pixelated" alt="CIFAR-10 diffusion samples enlarged with nearest-neighbor interpolation">

## Next Steps

The final written conclusions will be added after PCA results are merged:

- PCA vs AE reconstruction quality on MNIST and Fashion-MNIST
- Metric trends by latent dimension or component count
- Where PCA is competitive
- Where AE/DAE/VAE models improve or fail
- Final limitations and recommendations

## How To Reproduce

Autoencoder examples:

```bash
python train.py --model ae --dataset mnist --latent-dim 16
python train.py --model dae --dataset mnist --latent-dim 16 --dae-noise-level 0.2
python train.py --model vae --dataset mnist --latent-dim 16
```

Fashion-MNIST is supported for the autoencoder side:

```bash
python train.py --model ae --dataset fashion --latent-dim 16
```

Diffusion extra-credit configs:

```bash
python train.py --config configs/diffusion/mnist.yaml
python train.py --config configs/diffusion/cifar10.yaml
```

Collect small report images into `docs/assets/`:

```bash
python scripts/collect_report_assets.py
```
