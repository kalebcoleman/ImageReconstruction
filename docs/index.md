# PCA vs Autoencoders for Image Reconstruction

## Overview

This project compares dimensionality reduction and reconstruction approaches across MNIST, Fashion-MNIST, and CIFAR-10. The main comparison is between PCA as a classical linear baseline and autoencoders as a neural-network reconstruction approach.

The core analysis is PCA vs autoencoder reconstruction quality. Variational autoencoders and diffusion are included as generative extensions: the VAE bridges reconstruction and sampling through a learned latent distribution, while the diffusion model explores modern image generation beyond the main reconstruction task.

## Main Research Question

How do PCA and autoencoder-based methods compare for image reconstruction quality across MNIST, Fashion-MNIST, and CIFAR-10?

## Project Roles

- **PCA analysis**: teammate
- **Autoencoder, VAE, and diffusion extension**: me

## Datasets

- **MNIST**: 28x28 grayscale handwritten digits.
- **Fashion-MNIST**: 28x28 grayscale clothing images.
- **CIFAR-10**: 32x32 RGB natural images with more complex objects, color, and background variation.

## Main Models

### PCA

PCA is the classical baseline for this project. It reconstructs images by projecting them into a lower-dimensional linear subspace and then mapping them back to image space.

**Placeholder for teammate's PCA results and analysis:**

- PCA reconstruction grids by dataset
- PCA metrics by latent dimension or number of principal components
- PCA qualitative observations
- PCA strengths and weaknesses compared with autoencoders

### Regular Autoencoder

The regular autoencoder is the main neural reconstruction model. It learns an encoder that compresses each image into a latent representation and a decoder that reconstructs the image from that compressed code.

The AE experiments focus on reconstruction quality, latent-space compression, and metrics such as MSE, PSNR, and SSIM. The key question is whether a nonlinear learned representation can preserve image structure better than PCA at similar latent dimensionalities.

### Variational Autoencoder

The VAE extends the autoencoder approach by learning a structured probabilistic latent space. It supports reconstruction, latent sampling, and interpolation experiments.

In this project, the VAE acts as a bridge between reconstruction and generative modeling. It still reconstructs input images, but it also makes it possible to sample new images and interpolate between latent codes to visualize how the learned representation changes smoothly.

## Main PCA vs Autoencoder Analysis

This section is reserved for the final merged comparison between the PCA baseline and the autoencoder results.

### Reconstruction Comparison Tables

**Placeholder:** add tables comparing PCA, AE, and VAE reconstruction quality across MNIST, Fashion-MNIST, and CIFAR-10.

### SSIM / PSNR / MSE Results

**Placeholder:** add metric summaries for each dataset and latent dimension.

### Latent Dimension Comparisons

**Placeholder:** compare how PCA and autoencoder reconstructions change as the latent dimension or number of PCA components increases.

### Visual Reconstruction Grids

**Placeholder:** add side-by-side grids showing original images, PCA reconstructions, AE reconstructions, and VAE reconstructions.

### Key Takeaways from PCA vs AE

**Placeholder:** summarize where PCA is competitive, where autoencoders improve reconstruction, and how dataset complexity changes the comparison.

## Extra Credit / Extension: Diffusion Model

Diffusion was added after the main reconstruction work as an extra-credit generative extension. It is not the main project and should be interpreted separately from the PCA vs autoencoder reconstruction comparison.

Instead of only reconstructing an input image, the diffusion model explores image generation through iterative denoising. This helped connect the reconstruction models in the main project to modern generative modeling.

MNIST and Fashion-MNIST generations worked better because those images are simpler: they are grayscale, low resolution, and have less variation than natural images. CIFAR-10 was much harder because it has RGB color, more complex objects, backgrounds, and greater variation across classes.

CIFAR generations should be shown at actual `32x32` size and also enlarged using nearest-neighbor interpolation. The actual-size view preserves the true model output, while nearest-neighbor enlargement keeps the pixels crisp instead of making the image look blurry from smoothing.

The diffusion model was intentionally lightweight, so imperfect CIFAR-10 results are expected. Stronger CIFAR generation would require more model capacity, longer training, and more compute.

## Diffusion Visualizations

Add diffusion outputs here after running the report asset preparation script.

### MNIST Generations

**Placeholder:** `results/mnist/`

### Fashion-MNIST Generations

**Placeholder:** `results/fashion_mnist/`

### CIFAR Actual-Size Generations

**Placeholder:** `results/cifar10/generations_actual_size.png`

### CIFAR Nearest-Neighbor Enlarged Generations

**Placeholder:** `results/cifar10/generations_scaled_nearest.png`

## Limitations

- PCA results still need to be merged from the teammate's analysis.
- CIFAR-10 is more difficult for small models because it contains RGB color, complex object structure, and high variation.
- The diffusion extension needs a larger UNet, longer training, and more compute for stronger CIFAR-10 generations.
- The final report depends on adding the missing PCA tables and side-by-side PCA vs AE visuals.

## Future Work

- Finish the PCA vs AE comparison tables.
- Add better side-by-side visual comparisons for PCA, AE, and VAE reconstructions.
- Improve diffusion with a larger UNet, attention blocks, a cosine noise schedule, EMA weights, and Monsoon GPU sweeps.
- Expand hyperparameter sweeps across latent dimensions, datasets, seeds, and training lengths.

## How to Run

The repo supports AE, DAE, VAE, and diffusion runs through `train.py`. Example commands:

```bash
# Regular autoencoder reconstruction
python train.py --model ae --dataset mnist --latent-dim 16

# Variational autoencoder reconstruction, generation, and interpolation
python train.py --model vae --dataset mnist --latent-dim 16

# Diffusion extension using existing configs
python train.py --config configs/diffusion/mnist.yaml
python train.py --config configs/diffusion/fashion.yaml
python train.py --config configs/diffusion/cifar10.yaml
```

Prepare report assets:

```bash
python scripts/prepare_report_assets.py
```

Preview without copying:

```bash
python scripts/prepare_report_assets.py --dry-run
```

Replace existing report assets intentionally:

```bash
python scripts/prepare_report_assets.py --overwrite
```
