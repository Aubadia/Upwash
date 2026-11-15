#!/usr/bin/env bash
#SBATCH --job-name=base150m-calibration
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=/work/asciibase64/upwash/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export WANDB_MODE=online
export TORCHINDUCTOR_COMPILE_THREADS=1
RUN_LOG=/work/asciibase64/upwash/logs/train-${SLURM_JOB_ID}.log
bash scripts/train.sbatch 2>&1 | tee "$RUN_LOG"
.venv/bin/python scripts/base_calibration_results.py collect "$RUN_LOG"
