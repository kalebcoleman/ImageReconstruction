#!/bin/bash
set -euo pipefail

# Submit the final MNIST/Fashion training jobs, then run all evaluations and
# refresh the final deliverables. Set CLEAR_INCOMPLETE=1 when rerunning after a
# failed or time-limited partial run.

: "${CLEAR_INCOMPLETE:=0}"
: "${ALLOW_MODEL_DOWNLOAD:=0}"
: "${DRY_RUN:=0}"
: "${SEEDS:=1 2 3}"

export CLEAR_INCOMPLETE
export ALLOW_MODEL_DOWNLOAD
export SEEDS

submit() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN submit:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

echo "Submitting MNIST training array with SEEDS=${SEEDS}"
submit sbatch slurm/final_study/train_mnist_array.slurm

echo "Submitting FashionMNIST training array with SEEDS=${SEEDS}; CLEAR_INCOMPLETE=${CLEAR_INCOMPLETE}"
submit sbatch slurm/final_study/train_fashion_array.slurm

cat <<'EOF'

After both training arrays finish, submit evaluation and finalization:

  sbatch slurm/final_study/eval_all_array.slurm
  sbatch slurm/final_study/finalize.slurm

If this is the first evaluation on the cluster and metric weights are not cached:

  ALLOW_MODEL_DOWNLOAD=1 sbatch slurm/final_study/eval_all_array.slurm
EOF
