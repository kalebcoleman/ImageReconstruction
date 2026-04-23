# Diffusion Parity Recipes

These recipe files lock a fair cross-dataset comparison protocol for the pixel-space ADM diffusion path.

Shared across `mnist_64.yaml`, `fashion_64.yaml`, `cifar10_64.yaml`, and `imagenet_64.yaml`:

- `image_size = 64`
- `diffusion_channels = 3`
- `diffusion_backbone = adm`
- `diffusion_preprocessing = parity_64`
- `prediction_type = v`
- `schedule = cosine`
- `ema_decay = 0.999`
- `class_dropout_prob = 0.1`
- `sampler = ddim`
- `sampling_steps = 50`
- `ddim_eta = 0.0`
- `attention_resolutions = [16, 8]`

Allowed to differ across datasets:

- `epochs`
- `batch_size`
- `num_workers`
- `eval_batch_size`
- `data_dir`
- `output_dir`
- `run_name`
- the concrete ImageNet subset/full-data realization behind `data_dir`

`imagenet_64.yaml` is a protocol definition, not a claim that the repo has already validated a full ImageNet-64 benchmark.

Lightweight smoke-test variants live under `smoke/` with the same dataset scope and preprocessing path, but reduced runtime defaults intended for quick end-to-end validation of the parity-study pipeline.
