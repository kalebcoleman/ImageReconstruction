# Deliverables Index

Use this folder as the presentation surface. The training code and study
orchestration stay in the repo root; final outputs live here.

## What To Show

- `presentation_crisp/`: nearest-neighbor upscaled sample grids for slides.
  Use the `*_4x_nearest.png` files when you want the image to stay crisp while
  zooming. These files preserve the generated pixels instead of letting the
  viewer blur them.
- `presentation_compact/`: research-style CIFAR10 contact sheets. Use these
  when you want many samples shown at their natural size instead of enlarged
  individual tiles. The `*_native_1x.png` files are true native-size grids; the
  `*_native_2x.png` files stay compact but are easier to place in slides.
- `image-reconstruction-final-study/runs/mnist/diffusion/`: completed MNIST
  diffusion runs with checkpoints, configs, metrics, manifests, and samples.
- `image-reconstruction-final-study/runs/cifar10/diffusion/`: completed CIFAR10
  diffusion runs with checkpoints, configs, metrics, manifests, and samples.
- `image-reconstruction-final-study/runs/fashion/diffusion/`: FashionMNIST run
  folders. These currently need a rerun because the copied logs stop before
  final checkpoints and artifacts are written.
- `cifar10-smoke/`: smoke-test outputs only. Do not use these as final quality
  figures.

## Final Model Locations

Completed final-study checkpoints follow this pattern:

```text
deliverables/image-reconstruction-final-study/runs/<dataset>/diffusion/study_<dataset>_seed<seed>/checkpoints/best.pt
```

Examples:

```text
deliverables/image-reconstruction-final-study/runs/mnist/diffusion/study_mnist_seed001/checkpoints/best.pt
deliverables/image-reconstruction-final-study/runs/cifar10/diffusion/study_cifar10_seed001/checkpoints/best.pt
```

Each completed run also has:

```text
config.yaml
metrics.json
metrics.jsonl
run_manifest.md
samples/generated_samples.png
samples/generated_samples_native_grid.png
plots/reconstructions.png
plots/diffusion_snapshots.png
plots/loss_curve.png
```

## CIFAR10 Display Notes

CIFAR10 is native `32x32`, so zooming into generated samples will always expose
the low pixel count. For slides, use nearest-neighbor presentation copies so the
pixels stay sharp instead of becoming browser-smoothed blur.

For a paper-style view that does not zoom individual images, use:

```text
deliverables/presentation_compact/cifar10_seed001_contact_100_native_2x.png
deliverables/presentation_compact/cifar10_seed002_contact_100_native_2x.png
deliverables/presentation_compact/cifar10_seed003_contact_100_native_2x.png
```

New diffusion training and checkpoint-evaluation runs also save a compact
native grid automatically as:

```text
samples/generated_samples_native_grid.png
artifacts/generated_samples_native_grid.png
```

For genuinely larger generated outputs, train a larger recipe instead of only
resizing an existing PNG:

```bash
python train.py \
  --config configs/diffusion/cifar10_showcase64.yaml \
  --data-dir /scratch/$USER/image-reconstruction/data \
  --output-dir /scratch/$USER/image-reconstruction-showcase \
  --run-name cifar10_showcase64_seed001 \
  --seed 1
```

The 64x64 showcase recipe is intended for presentation. It is more expensive
than the default 32x32 final-study CIFAR10 recipe and should be reported
separately from the native-resolution study.

## FashionMNIST Rerun

The copied FashionMNIST logs stop before final artifacts were saved. Rerun with
incomplete outputs cleared:

```bash
CLEAR_INCOMPLETE=1 sbatch slurm/final_study/train_fashion_array.slurm
```

After Fashion training finishes, run evaluation and finalization:

```bash
sbatch slurm/final_study/eval_all_array.slurm
sbatch slurm/final_study/finalize.slurm
```
