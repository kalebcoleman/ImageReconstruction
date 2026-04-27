# Image Reconstruction

This project compares PCA and autoencoder-based methods for image
reconstruction across MNIST, Fashion-MNIST, and CIFAR-10. PCA is the classical
baseline handled by my teammate; my side covers regular autoencoders, VAEs,
latent interpolation/generation experiments, and a diffusion model as an
extra-credit generative extension.

The final project writeup lives at [`docs/index.md`](docs/index.md).

The repo keeps the original `ae`, `dae`, and `vae` paths, plus the
dataset-appropriate diffusion extension:

- `mnist`: legacy diffusion, native `28x28`, `1` channel
- `fashion`: legacy diffusion, native `28x28`, `1` channel
- `cifar10`: ADM diffusion, native `32x32`, `3` channels

ImageNet is not part of the default final study.

## Repository Structure

```text
.
├── diffusion/              # DDPM model, scheduler, training, and sampling code
├── scripts/                # Utility scripts for reports and visualizations
├── docs/                   # GitHub Pages project website
├── results/                # Final selected results and figures
├── archived_experiments/   # Older runs and exploratory outputs
├── train.py                # Main training entry point
└── README.md
```

## Main Entry Points

- [`train.py`](train.py): train one run from CLI flags or a config
- [`evaluate.py`](evaluate.py): checkpoint-only evaluation and sampling
- [`run_parity_suite.py`](run_parity_suite.py): final-study orchestration
- [`aggregate_results.py`](aggregate_results.py): manual aggregation helper

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

- [`configs/diffusion/mnist.yaml`](configs/diffusion/mnist.yaml)
- [`configs/diffusion/fashion.yaml`](configs/diffusion/fashion.yaml)
- [`configs/diffusion/cifar10.yaml`](configs/diffusion/cifar10.yaml)
- [`configs/diffusion/cifar10_showcase64.yaml`](configs/diffusion/cifar10_showcase64.yaml) — optional presentation recipe, not part of the default final study
- [`configs/diffusion/base_legacy28_gray.yaml`](configs/diffusion/base_legacy28_gray.yaml)
- [`configs/diffusion/base_adm32.yaml`](configs/diffusion/base_adm32.yaml)

Default full-study epochs:

- MNIST: `50` epochs in [`configs/diffusion/mnist.yaml`](configs/diffusion/mnist.yaml)
- FashionMNIST: `75` epochs in [`configs/diffusion/fashion.yaml`](configs/diffusion/fashion.yaml)
- CIFAR10: `150` epochs in [`configs/diffusion/cifar10.yaml`](configs/diffusion/cifar10.yaml)

Smoke configs:

- [`configs/diffusion/smoke/mnist.yaml`](configs/diffusion/smoke/mnist.yaml)
- [`configs/diffusion/smoke/fashion.yaml`](configs/diffusion/smoke/fashion.yaml)
- [`configs/diffusion/smoke/cifar10.yaml`](configs/diffusion/smoke/cifar10.yaml)
- [`configs/diffusion/smoke/base_legacy28_gray_smoke.yaml`](configs/diffusion/smoke/base_legacy28_gray_smoke.yaml)
- [`configs/diffusion/smoke/base_adm32_smoke.yaml`](configs/diffusion/smoke/base_adm32_smoke.yaml)

## Monsoon Commands

The recommended Monsoon path is the Slurm final-study array set in
[`slurm/final_study/`](slurm/final_study).
Dataset-specific training arrays are preferred because Slurm arrays share one
time limit: MNIST uses 8-hour jobs, FashionMNIST uses 16-hour jobs, and CIFAR10 uses 24-hour
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

Current MNIST/Fashion finish pass:

```bash
sbatch slurm/final_study/train_mnist_array.slurm
CLEAR_INCOMPLETE=1 sbatch slurm/final_study/train_fashion_array.slurm
sbatch slurm/final_study/eval_all_array.slurm
sbatch slurm/final_study/finalize.slurm
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
- MNIST training: A100, 8h, 4 CPUs, 32G
- FashionMNIST training: A100, 16h, 4 CPUs, 32G
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

Current copied deliverables are indexed in [`deliverables/README.md`](deliverables/README.md).
The GitHub Pages project writeup lives at [`docs/index.md`](docs/index.md).

## Final Project Website

The final project website writeup is [`docs/index.md`](docs/index.md). The main
project is a PCA vs autoencoder reconstruction comparison: PCA is the classical
dimensionality-reduction baseline, while the autoencoder side tests learned
nonlinear reconstruction with AE, DAE, and VAE experiments.

Diffusion is included as extra credit, not as the main project. It explores
generation after the reconstruction work and should be reported separately from
the PCA vs AE comparison.

Report assets for GitHub Pages can be copied into `docs/assets/` with:

```bash
python scripts/collect_report_assets.py
```

## GitHub Pages Setup

To publish the documentation site with GitHub Pages:

1. Go to **Settings → Pages** in the GitHub repository.
2. Set **Source** to **Deploy from branch**.
3. Set **Branch** to `main`.
4. Set **Folder** to `/docs`.
5. Click **Save**.

GitHub will then provide the website URL after the Pages build finishes.

## CIFAR10 Presentation Quality

CIFAR10 final-study samples are generated at native `32x32`. That is correct
for the study, but any slide viewer that smooths the pixels will make zoomed
images look blurry. Use nearest-neighbor upscaled copies for slides when you
want crisp pixels:

```text
deliverables/presentation_crisp/*_4x_nearest.png
```

For a research-paper-style view, use compact native contact sheets instead of
enlarged tiles:

```text
deliverables/presentation_compact/*_native_2x.png
```

New diffusion runs save this style automatically as
`generated_samples_native_grid.png` next to the normal generated sample grid.

Generating at `64x64` can make presentation images easier to view, but only if
the model is trained and sampled at `64x64`; resizing an existing `32x32` PNG
does not create real detail. The optional showcase recipe is:

```bash
python train.py \
  --config configs/diffusion/cifar10_showcase64.yaml \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --output-dir /scratch/$USER/image-reconstruction-showcase \
  --run-name cifar10_showcase64_seed001 \
  --seed 1
```

Treat that as a presentation/showcase run, not a replacement for the native
`32x32` final-study result.

## Notes

- If a dataset is missing locally, add `--download` on a login node so
  `train.py` can fetch it.
- If a previous failed run left an incomplete deterministic output directory,
  add `--clear-incomplete` before rerunning the same study command.
- The first full evaluation on a machine may need `--allow-model-download` once
  to cache metric weights.

## Manual Use

Manual training still works through [`train.py`](train.py).
The repo still keeps:

- AE/DAE/VAE functionality for MNIST/Fashion-style runs
- legacy diffusion support
- ADM diffusion support
- checkpoint-only evaluation via [`evaluate.py`](evaluate.py)

Example manual config runs:

```bash
python train.py --config configs/diffusion/mnist.yaml --data-dir /scratch/$USER/image-reconstruction/data
python train.py --config configs/diffusion/fashion.yaml --data-dir /scratch/$USER/image-reconstruction/data
python train.py --config configs/diffusion/cifar10.yaml --data-dir /scratch/$USER/image-reconstruction/data
```
