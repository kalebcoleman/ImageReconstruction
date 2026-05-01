#!/bin/bash
#SBATCH --job-name=final_smoke
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=slurm/final_study/logs/%x_%A_%a.out
#SBATCH --error=slurm/final_study/logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/projects/ImageReconstruction}"
COMMON_SH="${REPO_DIR}/slurm/final_study/common.sh"
if [[ ! -f "${COMMON_SH}" ]]; then
  echo "Unable to find final study common.sh at: ${COMMON_SH}" >&2
  echo "Set REPO_DIR to the ImageReconstruction checkout before submitting with sbatch." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${COMMON_SH}"

CURRENT_USER="${USER:-$(id -un)}"
final_study_set_defaults "/scratch/${CURRENT_USER}/image-reconstruction-final-study-smoke-all"
: "${DATASETS:=mnist cifar10}"
: "${SMOKE_SEED:=1}"

DATASET_WORDS="$(final_study_normalize_words "${DATASETS}")"
read -r -a DATASET_ARRAY <<< "${DATASET_WORDS}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#DATASET_ARRAY[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; valid range is 0-$(( ${#DATASET_ARRAY[@]} - 1 ))." >&2
  exit 2
fi
DATASET="${DATASET_ARRAY[$TASK_ID]}"
RUN_NAME="smoke_${DATASET}_seed$(printf "%03d" "${SMOKE_SEED}")"
RUN_DIR="${STUDY_DIR}/runs/${DATASET}/diffusion/${RUN_NAME}"

final_study_prepare_runtime
final_study_print_context "smoke-train"
echo "Dataset: ${DATASET}"
echo "Seed: ${SMOKE_SEED}"
echo "Run: ${RUN_NAME}"

if [[ -f "${RUN_DIR}/checkpoints/best.pt" && -f "${RUN_DIR}/metrics.json" ]]; then
  echo "Completed smoke run already exists: ${RUN_DIR}"
  exit 0
fi
if [[ -d "${RUN_DIR}" && "${CLEAR_INCOMPLETE}" == "1" ]]; then
  echo "Removing incomplete smoke run: ${RUN_DIR}"
  rm -rf "${RUN_DIR}"
elif [[ -d "${RUN_DIR}" ]]; then
  echo "Incomplete smoke run exists: ${RUN_DIR}" >&2
  echo "Set CLEAR_INCOMPLETE=1 to delete it before rerunning." >&2
  exit 1
fi

final_study_activate_conda
final_study_cd_repo

COMMAND=(
  "${PYTHON_BIN}" train.py
  --config "configs/diffusion/smoke/${DATASET}.yaml"
  --seed "${SMOKE_SEED}"
  --run-name "${RUN_NAME}"
  --data-dir "${DATA_DIR}"
  --output-dir "${STUDY_DIR}/runs"
)

final_study_run_command "${COMMAND[@]}"
