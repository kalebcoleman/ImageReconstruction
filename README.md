# Image Reconstruction

This repo keeps the original `ae`, `dae`, and `vae` paths, plus the final
dataset-appropriate diffusion study:

- `mnist`: legacy diffusion, native `28x28`, `1` channel
- `fashion`: legacy diffusion, native `28x28`, `1` channel
- `cifar10`: ADM diffusion, native `32x32`, `3` channels

ImageNet is not part of the default final study.

## Main Entry Points

- [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py): train one run from CLI flags or a config
- [`evaluate.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/evaluate.py): checkpoint-only evaluation and sampling
- [`run_parity_suite.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/run_parity_suite.py): final-study orchestration
- [`aggregate_results.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/aggregate_results.py): manual aggregation helper

## Final Study Design

- MNIST and FashionMNIST use the legacy grayscale diffusion path with native
  `28x28` images.
- CIFAR10 uses the ADM diffusion path with native `32x32` RGB images.
- The default study no longer forces identical image size, channels, or
  backbone across datasets.

Study configs:

- [`configs/diffusion/mnist.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/mnist.yaml)
- [`configs/diffusion/fashion.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/fashion.yaml)
- [`configs/diffusion/cifar10.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/cifar10.yaml)
- [`configs/diffusion/base_legacy28_gray.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/base_legacy28_gray.yaml)
- [`configs/diffusion/base_adm32.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/base_adm32.yaml)

Smoke configs:

- [`configs/diffusion/smoke/mnist.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/mnist.yaml)
- [`configs/diffusion/smoke/fashion.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/fashion.yaml)
- [`configs/diffusion/smoke/cifar10.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/cifar10.yaml)
- [`configs/diffusion/smoke/base_legacy28_gray_smoke.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/base_legacy28_gray_smoke.yaml)
- [`configs/diffusion/smoke/base_adm32_smoke.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/base_adm32_smoke.yaml)

## Monsoon Commands

Quick MNIST smoke:

```bash
python run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-smoke-mnist \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets mnist \
  --phase train
```

Quick Fashion smoke:

```bash
python run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-smoke-fashion \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets fashion \
  --phase train
```

Quick CIFAR10 smoke:

```bash
python run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-smoke-cifar10 \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets cifar10 \
  --phase train
```

All-dataset smoke:

```bash
python run_parity_suite.py run \
  --smoke \
  --study-dir /scratch/$USER/image-reconstruction-smoke-all \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --datasets mnist fashion cifar10 \
  --phase train
```

Full final study:

```bash
python run_parity_suite.py run \
  --study-dir /scratch/$USER/image-reconstruction-final-study \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --phase both \
  --allow-model-download
```

Summaries and deliverables:

```bash
python run_parity_suite.py summarize \
  --study-dir /scratch/$USER/image-reconstruction-final-study

python run_parity_suite.py select-best \
  --study-dir /scratch/$USER/image-reconstruction-final-study

python run_parity_suite.py deliverables \
  --study-dir /scratch/$USER/image-reconstruction-final-study
```

## Notes

- If a dataset is missing locally, add `--download` on a login node so
  `train.py` can fetch it.
- If a previous failed run left an incomplete deterministic output directory,
  add `--clear-incomplete` before rerunning the same study command.
- The first full evaluation on a machine may need `--allow-model-download` once
  to cache metric weights.

## Manual Use

Manual training still works through [`train.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/train.py).
The repo still keeps:

- AE/DAE/VAE functionality for MNIST/Fashion-style runs
- legacy diffusion support
- ADM diffusion support
- checkpoint-only evaluation via [`evaluate.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/evaluate.py)

Example manual config runs:

```bash
python train.py --config configs/diffusion/mnist.yaml --data-dir /scratch/$USER/image-reconstruction/data
python train.py --config configs/diffusion/fashion.yaml --data-dir /scratch/$USER/image-reconstruction/data
python train.py --config configs/diffusion/cifar10.yaml --data-dir /scratch/$USER/image-reconstruction/data
```
