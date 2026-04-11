#!/bin/bash
#SBATCH --job-name=array_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-17%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "array_diffusion.slurm is an alias for the medium sweep."
echo "For the larger optional study, submit slurm/array_diffusion_large.slurm instead."

exec bash "${REPO_ROOT}/slurm/array_diffusion_medium.slurm"
