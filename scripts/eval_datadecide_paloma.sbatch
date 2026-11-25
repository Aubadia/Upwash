#!/usr/bin/env bash
#SBATCH --job-name=datadecide-paloma
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/asciibase64/upwash/datadecide-paloma/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
root=/work/asciibase64/upwash/datadecide-paloma
export HF_HOME="$root/cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=60
export TMPDIR="$root/tmp/$SLURM_JOB_ID"
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
exec "$root/olmo-venv/bin/python" scripts/datadecide_paloma.py score "$@"
