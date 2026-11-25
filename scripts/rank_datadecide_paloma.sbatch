#!/usr/bin/env bash
#SBATCH --job-name=datadecide-rank
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/work/asciibase64/upwash/datadecide-paloma/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
root=/work/asciibase64/upwash/datadecide-paloma
export TMPDIR="$root/tmp/$SLURM_JOB_ID"
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_datadecide_paloma.py
exec .venv/bin/python scripts/datadecide_paloma.py rank "$@"
