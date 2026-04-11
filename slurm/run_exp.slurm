#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=slurm/common.sh
source "${REPO_ROOT}/slurm/common.sh"

JOB_ID="${SLURM_JOB_ID:-manual}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-diffusion}"
DATASET="$(normalize_dataset_name "${DATASET:-mnist}")"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
LATENT_DIM="${LATENT_DIM:-8}"
TIMESTEPS="${TIMESTEPS:-250}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
TIME_DIM="${TIME_DIM:-64}"
BETA_START="${BETA_START:-1e-4}"
BETA_END="${BETA_END:-2e-2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12}"
NUM_WORKERS="${NUM_WORKERS:-$(default_num_workers 4)}"
DIFFUSION_LOG_INTERVAL="${DIFFUSION_LOG_INTERVAL:-25}"
DOWNLOAD="${DOWNLOAD:-0}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"

if [[ -z "${RUN_NAME:-}" ]]; then
  lr_tag="$(sanitize_tag "${LR}")"
  if [[ "${MODEL}" == "diffusion" ]]; then
    RUN_NAME="${DATASET}_${MODEL}_t${TIMESTEPS}_ch${BASE_CHANNELS}_bs${BATCH_SIZE}_lr${lr_tag}_seed${SEED}_job${JOB_ID}"
  else
    RUN_NAME="${DATASET}_${MODEL}_z${LATENT_DIM}_bs${BATCH_SIZE}_lr${lr_tag}_seed${SEED}_job${JOB_ID}"
  fi
fi

TASK_TMP_ROOT="${OUTPUT_DIR}/_slurm_tmp/${RUN_NAME}"

load_cluster_modules
activate_environment "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}" "${DATA_DIR}"
ensure_dataset_ready "${DATASET}" "${DATA_DIR}" "${DOWNLOAD}"
configure_python_runtime "${TASK_TMP_ROOT}" 4

echo "== Single experiment =="
echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "job_id: ${JOB_ID}"
echo "python: $(command -v "${PYTHON_BIN}")"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "dataset: ${DATASET}"
echo "model: ${MODEL}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "lr: ${LR}"
echo "seed: ${SEED}"
echo "num_workers: ${NUM_WORKERS}"
echo "timesteps: ${TIMESTEPS}"
echo "base_channels: ${BASE_CHANNELS}"
echo "time_dim: ${TIME_DIM}"
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
  --latent_dim "${LATENT_DIM}"
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

echo "Experiment completed. Expected run directory:"
echo "${OUTPUT_DIR}/${DATASET}/${MODEL}/${RUN_NAME}"
