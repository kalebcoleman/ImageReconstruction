# Slurm diffusion experiments

These Slurm scripts are designed for shared GPU clusters and follow the core HPC safety rules for coursework and demos:

- one training process per Slurm task
- one GPU per task
- no backgrounded Python jobs inside a single allocation
- unique run names and isolated output directories
- capped array concurrency for shared-cluster friendliness

All experiment artifacts land under:

```text
outputs/<dataset>/<model>/<run_name>/
```

Example:

```text
outputs/mnist/diffusion/mnist_diffusion_t500_ch16_bs128_lr1e-3_seed42_job12345_a8/
```

## Files

- `common.sh`: shared environment activation, dataset checks, runtime cache setup, and `srun` launch helper.
- `test_diffusion.slurm`: cheap smoke test that verifies environment activation, CUDA visibility, dataset access, CLI parsing, output writing, and logging.
- `run_exp.slurm`: single-run entrypoint for ad hoc experiments or follow-up reruns of one specific config.
- `array_diffusion_medium.slurm`: main report-friendly sweep. This is the best default for a class project demo.
- `array_diffusion_large.slurm`: optional heavier sweep that adds 1000-timestep runs and limited optimizer/batch sensitivity checks.
- `array_diffusion.slurm`: alias for the medium sweep so old submission habits still work.

## Environment activation

Each script looks for an environment in this order:

1. `VENV_DIR=/path/to/venv`
2. `./.venv`
3. `./venv`
4. `CONDA_ENV_NAME=<name>`

If your cluster requires modules, replace the commented `module load` lines in `slurm/common.sh`.

## Trainer CLI used by the scripts

The scripts now use the real trainer flags directly:

- `--model`
- `--dataset`
- `--epochs`
- `--batch_size`
- `--lr`
- `--seed`
- `--data_dir`
- `--output_dir`
- `--run_name`
- `--num_workers`
- `--download` / `--no-download`
- `--timesteps`
- `--base_channels`
- `--time_dim`
- `--diffusion_log_interval`

Backward-compatible aliases such as `--batch-size` and `--diffusion-timesteps` still work, but the Slurm scripts use the canonical names above.

## Recommended workflow

### 1. Smoke test

Use this first on any new cluster setup, environment, or filesystem path:

```bash
sbatch slurm/test_diffusion.slurm
```

Default smoke config:

- `epochs=1`
- `timesteps=20`
- `base_channels=8`
- `batch_size=32`
- `lr=1e-3`
- `seed=42`

Why it exists:

- catches bad env activation
- confirms the GPU is visible
- confirms the dataset is readable
- confirms the CLI writes a unique run directory and logs cleanly

### 2. Medium sweep

This is the main showable experiment set:

```bash
sbatch slurm/array_diffusion_medium.slurm
```

It runs 18 tasks total with `#SBATCH --array=0-17%3`:

- timesteps: `100`, `250`, `500`
- base channels: `8`, `16`, `32`
- seeds: `42`, `123`
- fixed batch size: `128`
- fixed learning rate: `1e-3`

Why this medium sweep is scientifically useful:

- it isolates the two most interpretable diffusion factors here: noise schedule length and model width
- it keeps batch size and learning rate fixed so comparisons are easier to explain in a report
- it includes a second seed, which lets you say the results are not based on a single lucky run
- it is broad enough to show tradeoffs without turning into a reckless Cartesian explosion

### 3. Large sweep

Use this only after the medium sweep behaves well:

```bash
sbatch slurm/array_diffusion_large.slurm
```

It runs 20 tasks total with `#SBATCH --array=0-19%2`:

- extends schedule length to `1000` timesteps
- keeps the stronger widths `16` and `32`
- keeps two seeds on the core `250/500/1000` schedule study
- adds a small set of `batch_size=64` and `lr=3e-4` runs on the expensive `500/1000` settings

Why this large sweep is still defensible:

- 1000 timesteps is expensive enough to be worth discussing explicitly
- the extra optimizer/batch checks are targeted at the high-cost settings instead of every combination
- the `%2` concurrency cap keeps the run polite on shared hardware

## Single-run entrypoint

Use `run_exp.slurm` to rerun one config cleanly:

```bash
sbatch --export=ALL,MODEL=diffusion,DATASET=mnist,EPOCHS=15,BATCH_SIZE=128,LR=1e-3,SEED=42,TIMESTEPS=250,BASE_CHANNELS=16,OUTPUT_DIR=/scratch/$USER/image-reconstruction/outputs,DATA_DIR=/shared/datasets/image-reconstruction slurm/run_exp.slurm
```

This is useful for:

- repeating the best medium config
- running a one-off sanity check
- generating one polished sample directory for a demo or screenshot

## Editing the sweep

Medium sweep:

- edit `TIMESTEPS_GRID`, `BASE_CHANNELS_GRID`, or `SEED_GRID` in [`array_diffusion_medium.slurm`](/Users/itzjuztmya/Kaleb/ImageReconstruction/slurm/array_diffusion_medium.slurm)
- if you change the total combinations, update `#SBATCH --array=...`

Large sweep:

- edit `TASK_SPECS` in [`array_diffusion_large.slurm`](/Users/itzjuztmya/Kaleb/ImageReconstruction/slurm/array_diffusion_large.slurm)
- each row is: `timesteps base_channels batch_size learning_rate seed`
- if you add or remove rows, update `#SBATCH --array=...`

## Runtime and cost notes

Relative cost climbs roughly with diffusion timestep count:

- `100` timesteps: cheap and useful for trend lines
- `250` timesteps: good middle ground for quality vs runtime
- `500` timesteps: strong report/demo setting
- `1000` timesteps: optional heavy setting and the one most likely to dominate wall time

Expect the large sweep to cost noticeably more than the medium sweep because:

- 1000-step sampling and training are substantially slower than 500-step runs
- larger-width models have more parameters
- reduced batch size can stretch epoch time on the expensive settings

## Dataset handling

By default the scripts require a pre-populated dataset cache:

```bash
DATA_DIR=/shared/datasets/image-reconstruction
```

If the dataset is missing and you explicitly want the job to download it, pass:

```bash
sbatch --export=ALL,DOWNLOAD=1 slurm/test_diffusion.slurm
```

For shared HPC systems, pre-downloading on a login node is still the better practice.
