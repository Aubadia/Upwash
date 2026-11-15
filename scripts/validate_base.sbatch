#!/usr/bin/env bash
#SBATCH --job-name=base150m-validate
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=/work/asciibase64/upwash/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export HF_HOME=/work/asciibase64/.cache/huggingface HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/work/asciibase64/upwash/cache/inductor/validation-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/work/asciibase64/upwash/cache/triton/validation-${SLURM_JOB_ID}
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
.venv/bin/python scripts/verify_base_data.py
.venv/bin/python tests/integration/validate_base_gpu.py
export CONFIG=configs/base150m/smoke.json RUN=base150m-fa3-compile-smoke-${SLURM_JOB_ID}
export OUT=/work/asciibase64/upwash/runs/$RUN WANDB_MODE=online
SMOKE_LOG=/work/asciibase64/upwash/logs/smoke-${SLURM_JOB_ID}.log
bash scripts/train.sbatch 2>&1 | tee "$SMOKE_LOG"
.venv/bin/python scripts/base_calibration_results.py launch "$SMOKE_LOG"
printf '%s\n' "$OUT" > /work/asciibase64/upwash/validation/smoke-output.txt
