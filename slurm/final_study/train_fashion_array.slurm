#!/bin/bash
#SBATCH --job-name=final_train_fashion
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=slurm/final_study/logs/%x_%A_%a.out
#SBATCH --error=slurm/final_study/logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CURRENT_USER="${USER:-$(id -un)}"
final_study_set_defaults "/scratch/${CURRENT_USER}/image-reconstruction-final-study"
: "${SEEDS:=1 2 3}"

SEED_WORDS="$(final_study_normalize_words "${SEEDS}")"
read -r -a SEED_ARRAY <<< "${SEED_WORDS}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SEED_ARRAY[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; valid range is 0-$(( ${#SEED_ARRAY[@]} - 1 ))." >&2
  exit 2
fi
SEED="${SEED_ARRAY[$TASK_ID]}"
DATASET="fashion"

final_study_prepare_runtime
final_study_print_context "train"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"

final_study_activate_conda
final_study_cd_repo

COMMAND=(
  "${PYTHON_BIN}" run_parity_suite.py run
  --study-dir "${STUDY_DIR}"
  --data-dir "${DATA_DIR}"
  --datasets "${DATASET}"
  --seeds "${SEED}"
  --phase train
  --skip-existing
  --no-summarize
)
if [[ "${CLEAR_INCOMPLETE}" == "1" ]]; then
  COMMAND+=(--clear-incomplete)
fi

final_study_run_command "${COMMAND[@]}"
