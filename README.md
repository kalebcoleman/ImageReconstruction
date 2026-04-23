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
- Presentation figure exports keep those native resolutions and render image
  grids with nearest-neighbor interpolation. This is especially important for
  CIFAR10: its native `32x32` images look blurry in slides if a viewer smooths
  them during display.

Study configs:

- [`configs/diffusion/mnist.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/mnist.yaml)
- [`configs/diffusion/fashion.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/fashion.yaml)
- [`configs/diffusion/cifar10.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/cifar10.yaml)
- [`configs/diffusion/base_legacy28_gray.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/base_legacy28_gray.yaml)
- [`configs/diffusion/base_adm32.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/base_adm32.yaml)

Default full-study epochs:

- MNIST: `50` epochs in [`configs/diffusion/mnist.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/mnist.yaml)
- FashionMNIST: `75` epochs in [`configs/diffusion/fashion.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/fashion.yaml)
- CIFAR10: `150` epochs in [`configs/diffusion/cifar10.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/cifar10.yaml)

Smoke configs:

- [`configs/diffusion/smoke/mnist.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/mnist.yaml)
- [`configs/diffusion/smoke/fashion.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/fashion.yaml)
- [`configs/diffusion/smoke/cifar10.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/cifar10.yaml)
- [`configs/diffusion/smoke/base_legacy28_gray_smoke.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/base_legacy28_gray_smoke.yaml)
- [`configs/diffusion/smoke/base_adm32_smoke.yaml`](/Users/itzjuztmya/Kaleb/ImageReconstruction/configs/diffusion/smoke/base_adm32_smoke.yaml)

## Monsoon Commands

The recommended Monsoon path is the Slurm final-study array set in
[`slurm/final_study/`](/Users/itzjuztmya/Kaleb/ImageReconstruction/slurm/final_study).
Dataset-specific training arrays are preferred because Slurm arrays share one
time limit: MNIST and FashionMNIST use 8-hour jobs, while CIFAR10 uses 24-hour
jobs.

```bash
sbatch slurm/final_study/smoke_array.slurm
```

Final training:

```bash
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_fashion_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

First evaluation if metric weights are not cached:

```bash
ALLOW_MODEL_DOWNLOAD=1 sbatch slurm/final_study/eval_all_array.slurm
```

Normal evaluation:

```bash
sbatch slurm/final_study/eval_all_array.slurm
```

Finalize summaries, selections, exported best artifacts, and deliverables:

```bash
sbatch slurm/final_study/finalize.slurm
```

Resource defaults:

- smoke: A100, 2h, 4 CPUs, 32G
- MNIST/Fashion training: A100, 8h, 4 CPUs, 32G
- CIFAR10 training: A100, 24h, 4 CPUs, 32G
- evaluation: A100, 12h, 4 CPUs, 32G

Logs are written under `slurm/final_study/logs/`.

Rerun failed arrays normally; scripts pass `--skip-existing`, so completed
deterministic outputs are reused:

```bash
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/eval_all_array.slurm
```

Rerun one failed task with `--array`; add `CLEAR_INCOMPLETE=1` only when a
failed task left an incomplete output directory:

```bash
sbatch --array=8 slurm/final_study/eval_all_array.slurm
CLEAR_INCOMPLETE=1 sbatch --array=8 slurm/final_study/eval_all_array.slurm
```

Use `sbatch --constraint=h200`, `--constraint=v100`, or
`--constraint=rtx6000` to switch GPU types. For a generic GPU, submit with an
empty constraint if your Slurm version accepts it, or remove the
`#SBATCH --constraint=a100` line from a local copy of the script.

The 1-epoch smoke outputs are expected to be blurry or noisy; they are pipeline
checks, not quality checks.

The deliverables and `export-best-artifacts` commands create normal figure
copies plus nearest-neighbor upscaled `*_presentation.png` copies by default.
Use `--no-presentation-copies` to skip them or `--presentation-scale 6` to
change the integer upscale factor.

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
