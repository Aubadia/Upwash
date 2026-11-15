#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?}"
export TORCHINDUCTOR_CACHE_DIR="/work/asciibase64/upwash/cache/speed150m/${SLURM_JOB_ID}/${LOCAL_RANK:-0}/inductor"
export TRITON_CACHE_DIR="/work/asciibase64/upwash/cache/speed150m/${SLURM_JOB_ID}/${LOCAL_RANK:-0}/triton"
export CUDA_CACHE_PATH="/work/asciibase64/upwash/cache/speed150m/${SLURM_JOB_ID}/${LOCAL_RANK:-0}/cuda"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec .venv/bin/python "$@"
