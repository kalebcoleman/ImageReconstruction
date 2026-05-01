# Diffusion

This package contains the extra-credit diffusion implementation:

- `model.py`: legacy MNIST-style UNet.
- `backbones/adm_unet.py`: ADM-style CIFAR10 UNet.
- `scheduler.py`: DDPM scheduler helpers.
- `sampling.py`: DDPM/DDIM sampling.
- `training.py`: diffusion train/eval loops and denoising metrics.
- `artifacts.py`: generated sample grids, reverse-process snapshots,
  reconstruction previews, and loss curves.

Use `train.py` as the CLI entry point:

```bash
python train.py --config configs/diffusion/mnist.yaml
python train.py --config configs/diffusion/cifar10.yaml
```
