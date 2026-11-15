#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?}"
export TORCHINDUCTOR_CACHE_DIR="/work/asciibase64/upwash/cache/speed150m/${SLURM_JOB_ID}/${LOCAL_RANK:-0}/inductor"
export TRITON_CACHE_DIR="/work/asciibase64/upwash/cache/speed150m/${SLURM_JOB_ID}/${LOCAL_RANK:-0}/triton"
export CUDA_CACHE_PATH="/work/asciibase64/upwash/cache/speed150m/${SLURM_JOB_ID}/${LOCAL_RANK:-0}/cuda"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
BENCH_ENTRY="/work/asciibase64/upwash/speed150m/${SLURM_JOB_ID}/bench.py"
if [[ ! -f "$BENCH_ENTRY" ]]; then BENCH_ENTRY=experiments/speed150m/bench.py; fi
exec .venv/bin/python "$BENCH_ENTRY" "$@"
