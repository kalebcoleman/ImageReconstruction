#!/bin/bash

default_storage_root() {
  local repo_root="$1"

  if [[ -n "${USER:-}" && -d "/scratch/${USER}" ]]; then
    printf '%s\n' "/scratch/${USER}/image-reconstruction"
    return
  fi

  if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" && -w "${SLURM_TMPDIR}" ]]; then
    printf '%s\n' "${SLURM_TMPDIR}/image-reconstruction"
    return
  fi

  printf '%s\n' "${repo_root}"
}


nearest_existing_parent() {
  local path="$1"
  local probe="${path}"

  while [[ ! -e "${probe}" && "${probe}" != "/" ]]; do
    probe="$(dirname "${probe}")"
  done

  printf '%s\n' "${probe}"
}


ensure_directory_ready() {
  local path="$1"
  local label="$2"

  if [[ -d "${path}" ]]; then
    return
  fi

  if [[ -e "${path}" && ! -d "${path}" ]]; then
    echo "${label} exists but is not a directory: ${path}" >&2
    exit 1
  fi

  local existing_parent
  existing_parent="$(nearest_existing_parent "${path}")"
  if [[ ! -w "${existing_parent}" ]]; then
    echo "${label} does not exist and cannot be created: ${path}" >&2
    echo "Nearest existing parent is not writable: ${existing_parent}" >&2
    echo "Set ${label^^} to a user-writable location such as /scratch/\$USER/image-reconstruction/..." >&2
    exit 1
  fi

  mkdir -p "${path}"
}

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
    cifar|cifar-10|cifar_10) printf '%s\n' "cifar10" ;;
    ilsvrc|ilsvrc2012) printf '%s\n' "imagenet" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

dataset_cache_dir_for() {
  case "$(normalize_dataset_name "$1")" in
    mnist) printf '%s\n' "MNIST" ;;
    fashion) printf '%s\n' "FashionMNIST" ;;
    cifar10) printf '%s\n' "cifar-10-batches-py" ;;
    imagenet) printf '%s\n' "imagenet" ;;
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

  ensure_directory_ready "${data_dir}" "Data directory"
  if [[ "${download}" == "1" ]]; then
    return
  fi

  local expected_dataset_dir
  expected_dataset_dir="$(dataset_cache_dir_for "${dataset}")"
  if [[ "$(normalize_dataset_name "${dataset}")" == "imagenet" ]]; then
    if [[ ! -d "${data_dir}/${expected_dataset_dir}/train" || ! -d "${data_dir}/${expected_dataset_dir}/val" ]]; then
      echo "ImageNet cache missing at ${data_dir}/${expected_dataset_dir}/{train,val}" >&2
      echo "Prepare the ImageNet train/ and val/ folders on shared storage before launching the job." >&2
      exit 1
    fi
    return
  fi

  if [[ ! -d "${data_dir}/${expected_dataset_dir}" ]]; then
    echo "Dataset cache missing at ${data_dir}/${expected_dataset_dir}" >&2
    echo "Populate the dataset on shared storage first, or resubmit with DOWNLOAD=1." >&2
    exit 1
  fi
}


ensure_output_directory_ready() {
  local output_dir="$1"
  ensure_directory_ready "${output_dir}" "Output directory"
}


ensure_log_directory_ready() {
  local log_dir="$1"
  ensure_directory_ready "${log_dir}" "Log directory"
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

  ensure_directory_ready "${task_tmp_root}" "Task temp directory"
  ensure_directory_ready "${task_tmp_root}/matplotlib" "Matplotlib cache directory"
  ensure_directory_ready "${task_tmp_root}/cache" "XDG cache directory"

  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${omp_threads}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${omp_threads}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${omp_threads}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${omp_threads}}"
  export MPLCONFIGDIR="${task_tmp_root}/matplotlib"
  export XDG_CACHE_HOME="${task_tmp_root}/cache"
  export TMPDIR="${SLURM_TMPDIR:-${task_tmp_root}}"

  ensure_directory_ready "${MPLCONFIGDIR}" "Matplotlib cache directory"
  ensure_directory_ready "${XDG_CACHE_HOME}" "XDG cache directory"
  ensure_directory_ready "${TMPDIR}" "Temporary directory"
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
