#!/bin/bash
#SBATCH --job-name=final_eval_all
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8
#SBATCH --output=slurm/final_study/logs/%x_%A_%a.out
#SBATCH --error=slurm/final_study/logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CURRENT_USER="${USER:-$(id -un)}"
final_study_set_defaults "/scratch/${CURRENT_USER}/image-reconstruction-final-study"
: "${DATASETS:=mnist fashion cifar10}"
: "${SEEDS:=1 2 3}"

DATASET_WORDS="$(final_study_normalize_words "${DATASETS}")"
SEED_WORDS="$(final_study_normalize_words "${SEEDS}")"
read -r -a DATASET_ARRAY <<< "${DATASET_WORDS}"
read -r -a SEED_ARRAY <<< "${SEED_WORDS}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET_COUNT="${#DATASET_ARRAY[@]}"
SEED_COUNT="${#SEED_ARRAY[@]}"
TASK_COUNT=$(( DATASET_COUNT * SEED_COUNT ))
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; valid range is 0-$(( TASK_COUNT - 1 ))." >&2
  exit 2
fi

DATASET_INDEX=$(( TASK_ID / SEED_COUNT ))
SEED_INDEX=$(( TASK_ID % SEED_COUNT ))
DATASET="${DATASET_ARRAY[$DATASET_INDEX]}"
SEED="${SEED_ARRAY[$SEED_INDEX]}"

final_study_prepare_runtime
final_study_print_context "eval"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "Allow model download: ${ALLOW_MODEL_DOWNLOAD}"

final_study_activate_conda
final_study_cd_repo

COMMAND=(
  "${PYTHON_BIN}" run_parity_suite.py run
  --study-dir "${STUDY_DIR}"
  --data-dir "${DATA_DIR}"
  --datasets "${DATASET}"
  --seeds "${SEED}"
  --phase eval
  --skip-existing
  --no-summarize
)
if [[ "${ALLOW_MODEL_DOWNLOAD}" == "1" ]]; then
  COMMAND+=(--allow-model-download)
fi
if [[ "${CLEAR_INCOMPLETE}" == "1" ]]; then
  COMMAND+=(--clear-incomplete)
fi

final_study_run_command "${COMMAND[@]}"
