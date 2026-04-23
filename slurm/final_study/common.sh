#!/bin/bash

final_study_set_defaults() {
  local default_study_dir="$1"
  local current_user="${USER:-$(id -un)}"

  : "${CONDA_ENV:=diffusion}"
  : "${REPO_DIR:=${HOME}/projects/ImageReconstruction}"
  : "${DATA_DIR:=/scratch/${current_user}/image-reconstruction/data}"
  : "${STUDY_DIR:=${default_study_dir}}"
  : "${GPU_CONSTRAINT:=a100}"
  : "${PARTITION:=gpu}"
  : "${ALLOW_MODEL_DOWNLOAD:=0}"
  : "${CLEAR_INCOMPLETE:=0}"
  : "${DRY_RUN:=0}"
  : "${PYTHON_BIN:=python}"
  : "${LOG_DIR:=${REPO_DIR}/slurm/final_study/logs}"
}

final_study_normalize_words() {
  local value="$1"
  value="${value//,/ }"
  printf '%s\n' "${value}"
}

final_study_prepare_runtime() {
  mkdir -p "${LOG_DIR}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${STUDY_DIR}"
  fi

  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local task_tmp_default
  if [[ "${DRY_RUN}" == "1" ]]; then
    task_tmp_default="${TMPDIR:-/tmp}/image-reconstruction-final-study-dry-run/${SLURM_JOB_ID:-local}_${task_id}"
  else
    task_tmp_default="${STUDY_DIR}/tmp/${SLURM_JOB_ID:-local}_${task_id}"
  fi
  : "${TASK_TMP_ROOT:=${SLURM_TMPDIR:-${task_tmp_default}}}"

  mkdir -p "${TASK_TMP_ROOT}/matplotlib"
  mkdir -p "${TASK_TMP_ROOT}/cache"

  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${TASK_TMP_ROOT}/matplotlib}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TASK_TMP_ROOT}/cache}"
  export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-${TASK_TMP_ROOT}}}"
}

final_study_activate_conda() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1; skipping conda activation."
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    return
  fi

  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    return
  fi

  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    return
  fi

  echo "Unable to find conda. Set CONDA_ENV and make conda available before submitting." >&2
  exit 1
}

final_study_cd_repo() {
  if [[ ! -d "${REPO_DIR}" ]]; then
    echo "Repository directory does not exist: ${REPO_DIR}" >&2
    echo "Set REPO_DIR to the ImageReconstruction checkout on Monsoon." >&2
    exit 1
  fi
  cd "${REPO_DIR}"
}

final_study_print_context() {
  local phase="$1"
  echo "Final study phase: ${phase}"
  echo "Job: ${SLURM_JOB_ID:-local}"
  echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}"
  echo "Repo: ${REPO_DIR}"
  echo "Study dir: ${STUDY_DIR}"
  echo "Data dir: ${DATA_DIR}"
  echo "Conda env: ${CONDA_ENV}"
  echo "Partition default/log value: ${PARTITION}"
  echo "GPU constraint default/log value: ${GPU_CONSTRAINT}"
  echo "Logs: ${LOG_DIR}"
}

final_study_run_command() {
  echo "Command:"
  printf '  %q' "$@"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --ntasks=1 "$@"
    return
  fi

  "$@"
}
