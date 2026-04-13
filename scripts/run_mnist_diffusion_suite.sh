#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

CONCURRENCY=3
NUM_WORKERS="${NUM_WORKERS:-0}"
SHORT_EPOCHS="${SHORT_EPOCHS:-15}"
FINAL_EPOCHS="${FINAL_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
TIMESTEPS="${TIMESTEPS:-500}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
TIME_DIM="${TIME_DIM:-64}"
SAMPLE_COUNT="${SAMPLE_COUNT:-12}"
DIFFUSION_LOG_INTERVAL="${DIFFUSION_LOG_INTERVAL:-25}"
WITH_FINAL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mnist_diffusion_suite.sh [options]

Options:
  --with-final            Run the long final job after the 4 short ablations finish
  --concurrency N         Max number of concurrent short runs (default: 3)
  --python PATH           Python executable to use (default: python3 or $PYTHON_BIN)
  --data-dir PATH         Dataset root (default: ./data)
  --output-dir PATH       Output root (default: ./outputs)
  --log-dir PATH          Wrapper log directory (default: ./logs)
  --num-workers N         Passed to train.py (default: 0)
  --short-epochs N        Epochs for the 4 short runs (default: 15)
  --final-epochs N        Epochs for the final long run (default: 60)
  --batch-size N          Batch size for all runs (default: 128)
  --lr VALUE              Learning rate for all runs (default: 1e-3)
  --seed N                Seed for all runs (default: 42)
  --timesteps N           Diffusion timesteps (default: 500)
  --base-channels N       Base channels (default: 64)
  --time-dim N            Time embedding dim (default: 64)
  --sample-count N        Samples saved per run (default: 12)
  --diffusion-log-interval N
                          Diffusion logging interval (default: 25)
  --help                  Show this message

Examples:
  bash scripts/run_mnist_diffusion_suite.sh
  bash scripts/run_mnist_diffusion_suite.sh --with-final --concurrency 3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-final)
      WITH_FINAL=1
      shift
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --short-epochs)
      SHORT_EPOCHS="$2"
      shift 2
      ;;
    --final-epochs)
      FINAL_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --timesteps)
      TIMESTEPS="$2"
      shift 2
      ;;
    --base-channels)
      BASE_CHANNELS="$2"
      shift 2
      ;;
    --time-dim)
      TIME_DIM="$2"
      shift 2
      ;;
    --sample-count)
      SAMPLE_COUNT="$2"
      shift 2
      ;;
    --diffusion-log-interval)
      DIFFUSION_LOG_INTERVAL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/train.py" ]]; then
  echo "train.py not found under ${REPO_ROOT}" >&2
  exit 1
fi

if ! [[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] || (( CONCURRENCY < 1 )); then
  echo "--concurrency must be an integer >= 1" >&2
  exit 1
fi

declare -a SHORT_RUN_NAMES=(
  "mnist_t${TIMESTEPS}_ch${BASE_CHANNELS}_linear_baseline_e${SHORT_EPOCHS}"
  "mnist_t${TIMESTEPS}_ch${BASE_CHANNELS}_linear_ema999_e${SHORT_EPOCHS}"
  "mnist_t${TIMESTEPS}_ch${BASE_CHANNELS}_cosine_e${SHORT_EPOCHS}"
  "mnist_t${TIMESTEPS}_ch${BASE_CHANNELS}_cosine_ema999_nr2_e${SHORT_EPOCHS}"
)

declare -a SHORT_RUN_EXTRA_ARGS=(
  "--schedule linear --ema_decay 0.0 --num_res_blocks 1"
  "--schedule linear --ema_decay 0.999 --num_res_blocks 1"
  "--schedule cosine --ema_decay 0.0 --num_res_blocks 1"
  "--schedule cosine --ema_decay 0.999 --num_res_blocks 2"
)

FINAL_RUN_NAME="mnist_t${TIMESTEPS}_ch${BASE_CHANNELS}_cosine_ema999_nr2_e${FINAL_EPOCHS}"
FINAL_RUN_EXTRA_ARGS="--schedule cosine --ema_decay 0.999 --num_res_blocks 2"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
declare -a FAILED_NAMES=()

build_train_command() {
  local run_name="$1"
  local epochs="$2"
  local extra_args="$3"

  printf '%q ' \
    "${PYTHON_BIN}" "${REPO_ROOT}/train.py" \
    --model diffusion \
    --dataset mnist \
    --epochs "${epochs}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${run_name}" \
    --num_workers "${NUM_WORKERS}" \
    --timesteps "${TIMESTEPS}" \
    --base_channels "${BASE_CHANNELS}" \
    --time_dim "${TIME_DIM}" \
    --sample_count "${SAMPLE_COUNT}" \
    --diffusion_log_interval "${DIFFUSION_LOG_INTERVAL}" \
    --no-download

  # shellcheck disable=SC2086
  printf '%s' "${extra_args}"
  printf '\n'
}

prune_finished_jobs() {
  local pid
  local name
  local idx
  local -a next_pids=()
  local -a next_names=()

  for idx in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[idx]}"
    name="${ACTIVE_NAMES[idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_names+=("${name}")
      continue
    fi

    if wait "${pid}"; then
      echo "[done] ${name}"
    else
      echo "[failed] ${name}" >&2
      FAILED_NAMES+=("${name}")
    fi
  done

  ACTIVE_PIDS=("${next_pids[@]}")
  ACTIVE_NAMES=("${next_names[@]}")
}

wait_for_available_slot() {
  while (( ${#ACTIVE_PIDS[@]} >= CONCURRENCY )); do
    sleep 2
    prune_finished_jobs
  done
}

launch_background_run() {
  local run_name="$1"
  local epochs="$2"
  local extra_args="$3"
  local log_path="${LOG_DIR}/${run_name}.runner.log"
  local cmd

  cmd="$(build_train_command "${run_name}" "${epochs}" "${extra_args}")"
  echo "[launch] ${run_name}"
  echo "         log: ${log_path}"
  echo "         cmd: ${cmd}"

  (
    cd "${REPO_ROOT}"
    # shellcheck disable=SC2086
    eval "${cmd}"
  ) >"${log_path}" 2>&1 &

  ACTIVE_PIDS+=("$!")
  ACTIVE_NAMES+=("${run_name}")
}

run_final_job() {
  local log_path="${LOG_DIR}/${FINAL_RUN_NAME}.runner.log"
  local cmd

  cmd="$(build_train_command "${FINAL_RUN_NAME}" "${FINAL_EPOCHS}" "${FINAL_RUN_EXTRA_ARGS}")"
  echo "[launch-final] ${FINAL_RUN_NAME}"
  echo "               log: ${log_path}"
  echo "               cmd: ${cmd}"

  (
    cd "${REPO_ROOT}"
    # shellcheck disable=SC2086
    eval "${cmd}"
  ) | tee "${log_path}"
}

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Short epochs: ${SHORT_EPOCHS}"
echo "Concurrency: ${CONCURRENCY}"
echo "With final: ${WITH_FINAL}"
echo

for idx in "${!SHORT_RUN_NAMES[@]}"; do
  wait_for_available_slot
  launch_background_run "${SHORT_RUN_NAMES[idx]}" "${SHORT_EPOCHS}" "${SHORT_RUN_EXTRA_ARGS[idx]}"
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  sleep 2
  prune_finished_jobs
done

if (( ${#FAILED_NAMES[@]} > 0 )); then
  echo
  echo "Short suite completed with failures:" >&2
  for name in "${FAILED_NAMES[@]}"; do
    echo "  - ${name}" >&2
  done
  exit 1
fi

echo
echo "Short suite completed successfully."

if (( WITH_FINAL == 1 )); then
  echo
  run_final_job
fi
