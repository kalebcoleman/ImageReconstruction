#!/bin/bash
#SBATCH --job-name=diffusion_medium
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
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
TIME_DIM="${TIME_DIM:-64}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-2e-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12}"
NUM_WORKERS="${NUM_WORKERS:-$(default_num_workers 4)}"
DIFFUSION_LOG_INTERVAL="${DIFFUSION_LOG_INTERVAL:-50}"
DOWNLOAD="${DOWNLOAD:-0}"
STORAGE_ROOT="${STORAGE_ROOT:-$(default_storage_root "${REPO_ROOT}")}"
DATA_DIR="${DATA_DIR:-${STORAGE_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/outputs}"

# Medium sweep design:
# - fixed batch size and learning rate keep the interpretation clean
# - timesteps and base_channels are the main scientific factors
# - two seeds provide a lightweight robustness check without exploding the grid
TIMESTEPS_GRID=(100 250 500)
BASE_CHANNELS_GRID=(8 16 32)
SEED_GRID=(42 123)

load_cluster_modules
activate_environment "${REPO_ROOT}"

ensure_log_directory_ready "${REPO_ROOT}/logs"
ensure_output_directory_ready "${OUTPUT_DIR}"
ensure_dataset_ready "${DATASET}" "${DATA_DIR}" "${DOWNLOAD}"

num_timesteps="${#TIMESTEPS_GRID[@]}"
num_channels="${#BASE_CHANNELS_GRID[@]}"
num_seeds="${#SEED_GRID[@]}"
total_jobs=$((num_timesteps * num_channels * num_seeds))

if (( ARRAY_ID < 0 || ARRAY_ID >= total_jobs )); then
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_ID} is outside the configured sweep size (${total_jobs})." >&2
  exit 1
fi

channel_index=$(( ARRAY_ID % num_channels ))
timestep_index=$(( (ARRAY_ID / num_channels) % num_timesteps ))
seed_index=$(( ARRAY_ID / (num_channels * num_timesteps) ))

TIMESTEPS="${TIMESTEPS_GRID[timestep_index]}"
BASE_CHANNELS="${BASE_CHANNELS_GRID[channel_index]}"
SEED="${SEED_GRID[seed_index]}"

lr_tag="$(sanitize_tag "${LR}")"
RUN_NAME="${DATASET}_${MODEL}_t${TIMESTEPS}_ch${BASE_CHANNELS}_bs${BATCH_SIZE}_lr${lr_tag}_seed${SEED}_job${JOB_ID}_a${ARRAY_ID}"
TASK_TMP_ROOT="${OUTPUT_DIR}/_slurm_tmp/${RUN_NAME}"
configure_python_runtime "${TASK_TMP_ROOT}" 4

echo "== Medium diffusion sweep task =="
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

echo "Medium sweep task completed. Expected run directory:"
echo "${OUTPUT_DIR}/${DATASET}/${MODEL}/${RUN_NAME}"
