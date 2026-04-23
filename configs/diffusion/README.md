# Diffusion Study Configs

Default final-study configs:

- `mnist.yaml`: legacy diffusion, native `28x28`, `1` channel
- `fashion.yaml`: legacy diffusion, native `28x28`, `1` channel
- `cifar10.yaml`: ADM diffusion, native `32x32`, `3` channels

Shared bases:

- `base_legacy28_gray.yaml`
- `base_adm32.yaml`

Smoke configs:

- `smoke/mnist.yaml`
- `smoke/fashion.yaml`
- `smoke/cifar10.yaml`
- `smoke/base_legacy28_gray_smoke.yaml`
- `smoke/base_adm32_smoke.yaml`

Design summary:

- MNIST and FashionMNIST stay grayscale and native-size.
- CIFAR10 stays native `32x32` RGB.
- ImageNet is not part of the default final study.
- The old strict `64x64` RGB parity configs were removed from the default repo
  workflow during cleanup.

The default [`run_parity_suite.py`](/Users/itzjuztmya/Kaleb/ImageReconstruction/run_parity_suite.py)
flow resolves configs from this directory and `smoke/`.
