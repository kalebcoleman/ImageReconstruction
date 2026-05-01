# Slurm Jobs

The cleaned repo keeps only the simple reproducible diffusion jobs used for the extra-credit extension.

Use:

```bash
sbatch slurm/final_study/smoke_array.slurm
sbatch slurm/final_study/train_mnist_array.slurm
sbatch slurm/final_study/train_cifar10_array.slurm
```

See [`final_study/README.md`](final_study/README.md) for environment variables, output paths, rerun commands, and GPU constraint examples.
