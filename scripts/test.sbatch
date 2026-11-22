#!/usr/bin/env bash
#SBATCH --job-name=upwash-tests
#SBATCH --account=ACD114087
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Run the CPU test suite on a compute node (AGENTS.md: compute-intensive work goes through Slurm).
#   sbatch scripts/test.sbatch                      # full suite
#   sbatch scripts/test.sbatch tests/unit/prep      # a subset; arguments go to ./test.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
echo "[$(date +%T)] branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) node=$(hostname)"
./test.sh -p no:cacheprovider "$@"
