#!/usr/bin/env bash
#SBATCH --job-name=sweep300m
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=/work/asciibase64/upwash/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${MANIFEST:?Set MANIFEST}"
: "${ACCEPT_RESULTS:?Set ACCEPT_RESULTS to the validated 300M acceptance directory}"
.venv/bin/python experiments/sweep300m/check_acceptance.py "$ACCEPT_RESULTS"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PWD:$PWD/src"
export HF_HUB_OFFLINE=1 HF_HOME=/work/asciibase64/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_SOCKET_IFNAME=lo
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://flock-api.cerulean.works}
export WANDB_MODE=online WANDB_DIR=/work/asciibase64/upwash/wandb
OUT=${OUT:-/work/asciibase64/upwash/runs/$(basename "$MANIFEST" .json)-${SLURM_JOB_ID}}
mkdir -p "$OUT"
cp "$MANIFEST" "$OUT/manifest.json"
cp source-manifest.json "$OUT/source-manifest.json"
exec .venv/bin/torchrun --standalone --nproc_per_node=8 --no-python /bin/bash \
    experiments/speed150m/entry_rank.sh experiments/sweep300m/train_grouped.py \
    --manifest "$MANIFEST" --result "$OUT/result.json" --output-dir "$OUT" \
    --scope objective --mode default --fused-adam --prefetch --compile-optimizer --online
