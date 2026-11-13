#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=upwash-train-test

source "$HOME/.bashrc"
set -euo pipefail

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

report_exit() {
    job_status=$?
    END_TIME=$(date +%s)
    echo "========================================"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total Cost: $((END_TIME - START_TIME)) s"
    echo "Exit Status: $job_status"
    echo "========================================"
    exit "$job_status"
}
trap report_exit EXIT

# Submit from the repository root with its existing .venv prepared.
cd "${SLURM_SUBMIT_DIR:?Submit this job from the Upwash repository root}"

# Export the project's shell-compatible .env settings to srun and the SDK.
if [[ ! -f .env ]]; then
    echo "Missing .env in the repository root; configure it from .env.example." >&2
    exit 1
fi
set -a
source .env
set +a

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1
export FINEWEB_TRAIN_ROOT="${FINEWEB_TRAIN_ROOT:-/work/$USER/fineweb-edu-llama3}"
export FINEWEB_DEV_ROOT="${FINEWEB_DEV_ROOT:-/work/$USER/fineweb-edu-dev-tokens}"
export HF_HOME="/work/$USER/.cache/huggingface"
export TMPDIR="/work/$USER/upwash/tmp/${SLURM_JOB_ID}"
export WANDB_DIR="/work/$USER/upwash/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$HF_HOME"

export WANDB_MODE=online
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://flock-api.cerulean.works}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the repository-root .env}"
export WANDB_API_KEY
export FLOCK_PROJECT="${FLOCK_PROJECT:-lab6-predictable-scaling}"
export FLOCK_RUN_NAME="${FLOCK_RUN_NAME:-fineweb-smoke-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)}"

srun ./test.sh \
    tests/unit/test_adaptor.py \
    tests/unit/components/optimizers/test_muon_adamw.py \
    tests/integration/slurm_train_llama.py \
    --basetemp "$TMPDIR/pytest" --tb=short -s
