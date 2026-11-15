#!/usr/bin/env bash
# Executed by torchrun inside train.sbatch, never directly on a login-node GPU.
set -euo pipefail
: "${SLURM_JOB_ID:?GPU training requires sbatch}"
: "${LOCAL_RANK:?This wrapper requires torchrun}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/work/$USER/upwash/cache/inductor}/job-${SLURM_JOB_ID}/rank-${LOCAL_RANK}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/$USER/upwash/cache/triton}/job-${SLURM_JOB_ID}/rank-${LOCAL_RANK}"
export CUDA_CACHE_PATH="/work/$USER/upwash/cache/cuda/job-${SLURM_JOB_ID}/rank-${LOCAL_RANK}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec .venv/bin/python -m upwash.train "$@"
