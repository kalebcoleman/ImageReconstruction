#!/bin/bash
#SBATCH --job-name=final_finalize
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/final_study/logs/%x_%j.out
#SBATCH --error=slurm/final_study/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CURRENT_USER="${USER:-$(id -un)}"
final_study_set_defaults "/scratch/${CURRENT_USER}/image-reconstruction-final-study"

final_study_prepare_runtime
final_study_print_context "finalize"

final_study_activate_conda
final_study_cd_repo

final_study_run_command "${PYTHON_BIN}" run_parity_suite.py summarize --study-dir "${STUDY_DIR}"
final_study_run_command "${PYTHON_BIN}" run_parity_suite.py select-best --study-dir "${STUDY_DIR}"
final_study_run_command "${PYTHON_BIN}" run_parity_suite.py export-best-artifacts --study-dir "${STUDY_DIR}"
final_study_run_command "${PYTHON_BIN}" run_parity_suite.py deliverables --study-dir "${STUDY_DIR}"
