# Diffusion Study Configs

Default final-study configs:

- `mnist.yaml`: legacy diffusion, native `28x28`, `1` channel
- `cifar10.yaml`: ADM diffusion, native `32x32`, `3` channels

Shared bases:

- `base_legacy28_gray.yaml`
- `base_adm32.yaml`

Smoke configs:

- `smoke/mnist.yaml`
- `smoke/cifar10.yaml`
- `smoke/base_legacy28_gray_smoke.yaml`
- `smoke/base_adm32_smoke.yaml`

Design summary:

- MNIST stays grayscale and native-size.
- CIFAR10 stays native `32x32` RGB.
- Fashion-MNIST is still available to the older AE/DAE/VAE code paths through
  `train.py`, but it is no longer a diffusion final-study target.
- ImageNet is not part of the default final study.
- The old strict `64x64` RGB comparison configs were removed from the default
  repo workflow during cleanup.

The Slurm scripts in `slurm/final_study/` call `train.py` directly with these
configs and save figures under the selected output directory.
