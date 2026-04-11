#!/bin/bash
#SBATCH --job-name=diffusion_large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-19%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=slurm/common.sh
source "${REPO_ROOT}/slurm/common.sh"

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-diffusion}"
DATASET="$(normalize_dataset_name "${DATASET:-mnist}")"
EPOCHS="${EPOCHS:-15}"
TIME_DIM="${TIME_DIM:-64}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-2e-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12}"
NUM_WORKERS="${NUM_WORKERS:-$(default_num_workers 4)}"
DIFFUSION_LOG_INTERVAL="${DIFFUSION_LOG_INTERVAL:-50}"
DOWNLOAD="${DOWNLOAD:-0}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"

# Large sweep design:
# - the first 12 tasks extend the medium study to 1000 diffusion steps and keep two seeds
# - the last 8 tasks probe optimization sensitivity on the expensive 500/1000-step settings
# - this is intentionally curated rather than a full Cartesian product
TASK_SPECS=(
  "250 16 128 1e-3 42"
  "250 16 128 1e-3 123"
  "500 16 128 1e-3 42"
  "500 16 128 1e-3 123"
  "1000 16 128 1e-3 42"
  "1000 16 128 1e-3 123"
  "250 32 128 1e-3 42"
  "250 32 128 1e-3 123"
  "500 32 128 1e-3 42"
  "500 32 128 1e-3 123"
  "1000 32 128 1e-3 42"
  "1000 32 128 1e-3 123"
  "500 16 64 1e-3 42"
  "500 16 64 3e-4 42"
  "500 32 64 1e-3 42"
  "500 32 64 3e-4 42"
  "1000 16 64 1e-3 42"
  "1000 16 64 3e-4 42"
  "1000 32 64 1e-3 42"
  "1000 32 64 3e-4 42"
)

load_cluster_modules
activate_environment "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}" "${DATA_DIR}"
ensure_dataset_ready "${DATASET}" "${DATA_DIR}" "${DOWNLOAD}"

total_jobs="${#TASK_SPECS[@]}"
if (( ARRAY_ID < 0 || ARRAY_ID >= total_jobs )); then
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_ID} is outside the configured sweep size (${total_jobs})." >&2
  exit 1
fi

read -r TIMESTEPS BASE_CHANNELS BATCH_SIZE LR SEED <<< "${TASK_SPECS[ARRAY_ID]}"

lr_tag="$(sanitize_tag "${LR}")"
RUN_NAME="${DATASET}_${MODEL}_t${TIMESTEPS}_ch${BASE_CHANNELS}_bs${BATCH_SIZE}_lr${lr_tag}_seed${SEED}_job${JOB_ID}_a${ARRAY_ID}"
TASK_TMP_ROOT="${OUTPUT_DIR}/_slurm_tmp/${RUN_NAME}"
configure_python_runtime "${TASK_TMP_ROOT}" 4

echo "== Large diffusion sweep task =="
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
echo "num_workers: ${NUM_WORKERS}"
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
  --beta_start "${BETA_START}"
  --beta_end "${BETA_END}"
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

echo "Large sweep task completed. Expected run directory:"
echo "${OUTPUT_DIR}/${DATASET}/${MODEL}/${RUN_NAME}"
