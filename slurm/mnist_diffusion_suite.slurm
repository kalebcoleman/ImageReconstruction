#!/bin/bash
#SBATCH --job-name=mnist_diff_suite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
# shellcheck source=slurm/common.sh
source "${REPO_ROOT}/slurm/common.sh"

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-diffusion}"
DATASET="$(normalize_dataset_name "${DATASET:-mnist}")"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
TIMESTEPS="${TIMESTEPS:-500}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
TIME_DIM="${TIME_DIM:-64}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-2e-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12}"
NUM_WORKERS="${NUM_WORKERS:-$(default_num_workers 4)}"
DIFFUSION_LOG_INTERVAL="${DIFFUSION_LOG_INTERVAL:-25}"
DOWNLOAD="${DOWNLOAD:-0}"
STORAGE_ROOT="${STORAGE_ROOT:-$(default_storage_root "${REPO_ROOT}")}"
DATA_DIR="${DATA_DIR:-${STORAGE_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/outputs}"

# Task rows:
# run_name_suffix schedule ema_decay num_res_blocks
TASK_SPECS=(
  "linear_baseline linear 0.0 1"
  "linear_ema999 linear 0.999 1"
  "cosine cosine 0.0 1"
  "cosine_ema999_nr2 cosine 0.999 2"
)

load_cluster_modules
activate_environment "${REPO_ROOT}"

ensure_log_directory_ready "${REPO_ROOT}/logs"
ensure_output_directory_ready "${OUTPUT_DIR}"
ensure_dataset_ready "${DATASET}" "${DATA_DIR}" "${DOWNLOAD}"

total_jobs="${#TASK_SPECS[@]}"
if (( ARRAY_ID < 0 || ARRAY_ID >= total_jobs )); then
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_ID} is outside the configured suite size (${total_jobs})." >&2
  exit 1
fi

read -r VARIANT_TAG SCHEDULE EMA_DECAY NUM_RES_BLOCKS <<< "${TASK_SPECS[ARRAY_ID]}"

RUN_NAME="${DATASET}_t${TIMESTEPS}_ch${BASE_CHANNELS}_${VARIANT_TAG}_e${EPOCHS}_seed${SEED}_job${JOB_ID}_a${ARRAY_ID}"
TASK_TMP_ROOT="${OUTPUT_DIR}/_slurm_tmp/${RUN_NAME}"
configure_python_runtime "${TASK_TMP_ROOT}" 4

echo "== MNIST diffusion suite task =="
echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "job_id: ${JOB_ID}"
echo "array_id: ${ARRAY_ID}/${total_jobs}"
echo "python: $(command -v "${PYTHON_BIN}")"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "dataset: ${DATASET}"
echo "model: ${MODEL}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "lr: ${LR}"
echo "seed: ${SEED}"
echo "timesteps: ${TIMESTEPS}"
echo "base_channels: ${BASE_CHANNELS}"
echo "time_dim: ${TIME_DIM}"
echo "schedule: ${SCHEDULE}"
echo "ema_decay: ${EMA_DECAY}"
echo "num_res_blocks: ${NUM_RES_BLOCKS}"
echo "num_workers: ${NUM_WORKERS}"
echo "data_dir: ${DATA_DIR}"
echo "output_dir: ${OUTPUT_DIR}"
echo "run_name: ${RUN_NAME}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi

cmd=(
  "${PYTHON_BIN}" "${REPO_ROOT}/train.py"
  --model "${MODEL}"
  --dataset "${DATASET}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --seed "${SEED}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --num_workers "${NUM_WORKERS}"
  --timesteps "${TIMESTEPS}"
  --base_channels "${BASE_CHANNELS}"
  --time_dim "${TIME_DIM}"
  --schedule "${SCHEDULE}"
  --beta_start "${BETA_START}"
  --beta_end "${BETA_END}"
  --ema_decay "${EMA_DECAY}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --sample_count "${SAMPLE_COUNT}"
  --diffusion_log_interval "${DIFFUSION_LOG_INTERVAL}"
)

if [[ "${DOWNLOAD}" == "1" ]]; then
  cmd+=(--download)
else
  cmd+=(--no-download)
fi

echo "== Python command =="
print_command "${cmd[@]}"

run_with_srun "${cmd[@]}"

echo "Suite task completed. Expected run directory:"
echo "${OUTPUT_DIR}/${DATASET}/${MODEL}/${RUN_NAME}"
