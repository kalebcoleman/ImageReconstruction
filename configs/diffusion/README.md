# Diffusion Study Recipes

Default final-study recipes are dataset-appropriate instead of forcing every
dataset through the same `64x64` RGB ADM path.

Default full-study recipes:

- `mnist.yaml`: legacy diffusion, `28x28`, `1` channel, grayscale preprocessing
- `fashion.yaml`: legacy diffusion, `28x28`, `1` channel, grayscale preprocessing
- `cifar10.yaml`: ADM diffusion, `64x64`, `3` channels, natural-image preprocessing

Default smoke recipes live under `smoke/` with the same dataset choices and
reduced runtime defaults for quick end-to-end validation.

Key design change:

- MNIST and FashionMNIST are no longer converted to RGB `64x64` by default.
- CIFAR10 remains the primary ADM `64x64` RGB recipe.
- The old strict `64x64` RGB parity family is archived under `experimental/`
  and `smoke/experimental/` for comparison or reproduction only.

Recommended final-study entrypoints:

- `configs/diffusion/mnist.yaml`
- `configs/diffusion/fashion.yaml`
- `configs/diffusion/cifar10.yaml`

Recommended smoke entrypoints:

- `configs/diffusion/smoke/mnist.yaml`
- `configs/diffusion/smoke/fashion.yaml`
- `configs/diffusion/smoke/cifar10.yaml`

The default [`run_parity_suite.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/run_parity_suite.py)
flow resolves recipes from this directory and `smoke/`. It does not pick the
archived `experimental/` recipes unless you explicitly point `--config-dir`
there for comparison work.
