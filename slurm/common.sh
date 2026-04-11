#!/bin/bash

activate_environment() {
  local repo_root="$1"

  if [[ -n "${VENV_DIR:-}" && -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    return
  fi

  if [[ -f "${repo_root}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/.venv/bin/activate"
    return
  fi

  if [[ -f "${repo_root}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/venv/bin/activate"
    return
  fi

  if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    return
  fi

  echo "No Python environment found. Set VENV_DIR or CONDA_ENV_NAME for this cluster." >&2
  exit 1
}

load_cluster_modules() {
  if type module >/dev/null 2>&1; then
    module purge
    # TODO: Replace with the module stack for your cluster, for example:
    # module load cuda/12.2
    # module load python/3.11
  fi
}

normalize_dataset_name() {
  case "$1" in
    fashion-mnist|fashion_mnist) printf '%s\n' "fashion" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

dataset_cache_dir_for() {
  case "$(normalize_dataset_name "$1")" in
    mnist) printf '%s\n' "MNIST" ;;
    fashion) printf '%s\n' "FashionMNIST" ;;
    *)
      echo "Unsupported dataset for cache preflight: $1" >&2
      exit 1
      ;;
  esac
}

ensure_dataset_ready() {
  local dataset="$1"
  local data_dir="$2"
  local download="${3:-0}"

  mkdir -p "${data_dir}"
  if [[ "${download}" == "1" ]]; then
    return
  fi

  local expected_dataset_dir
  expected_dataset_dir="$(dataset_cache_dir_for "${dataset}")"
  if [[ ! -d "${data_dir}/${expected_dataset_dir}" ]]; then
    echo "Dataset cache missing at ${data_dir}/${expected_dataset_dir}" >&2
    echo "Populate the dataset on shared storage first, or resubmit with DOWNLOAD=1." >&2
    exit 1
  fi
}

default_num_workers() {
  local fallback="${1:-4}"
  local cpus="${SLURM_CPUS_PER_TASK:-${fallback}}"
  if (( cpus <= 1 )); then
    printf '%s\n' "0"
  else
    printf '%s\n' "$((cpus - 1))"
  fi
}

sanitize_tag() {
  local value="$1"
  value="${value// /}"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value//,/}"
  value="${value//=/}"
  printf '%s\n' "${value}"
}

configure_python_runtime() {
  local task_tmp_root="$1"
  local omp_threads="${2:-4}"

  mkdir -p "${task_tmp_root}" "${task_tmp_root}/matplotlib" "${task_tmp_root}/cache"

  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${omp_threads}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${omp_threads}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${omp_threads}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${omp_threads}}"
  export MPLCONFIGDIR="${task_tmp_root}/matplotlib"
  export XDG_CACHE_HOME="${task_tmp_root}/cache"
  export TMPDIR="${SLURM_TMPDIR:-${task_tmp_root}}"

  mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_with_srun() {
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --ntasks=1 "$@"
    return
  fi
  "$@"
}
