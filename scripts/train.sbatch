#!/usr/bin/env bash
#SBATCH --job-name=upwash
#SBATCH --account=ACD115198
#SBATCH --partition=dev               # dev: <= 4 h; longer runs: --partition=8gpus (<= 48 h)
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96            # Nano4 grants 12 cores per GPU
#SBATCH --time=00:30:00
# Nodes that failed in lab5/midterm (GPUs unopenable, /dev/shm full, slow, NCCL).
# Passing --exclude on the command line REPLACES this list, so repeat it.
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

# Usage, from the repository root (mkdir -p logs once):
#   WANDB_MODE=offline CONFIG=configs/smoke-150m.json sbatch -J smoke scripts/train.sbatch
# Keep smoke/debug runs offline: lab6-predictable-scaling is shared across pairs.
# The run summary still prints at the end of the Slurm log.
#   CONFIG=configs/sweep-150m-base.json RUN=lr2e-3 sbatch -J lr2e-3 --partition=8gpus --time=10:00:00 scripts/train.sbatch
# RUN names the Flock run and the export directory; it defaults to <config>-<job id>.
# One-time on the login node:
#   uv run --project /work/$USER/flock flock login            (also logs wandb in to Flock)
#   HF_HOME=/work/$USER/.cache/huggingface .venv/bin/python -c \
#     "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', revision='4e20de362430cd3b72f300e6b0f18e50e7166e08')"

JOB_START=$(date +%s)
log() { echo "[$(date +%H:%M:%S) +$(( $(date +%s) - JOB_START ))s] $*"; }

cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
CONFIG=${CONFIG:?Set CONFIG to a JSON run config}
RUN=${RUN:-$(basename "$CONFIG" .json)-${SLURM_JOB_ID}}
OUT=${OUT:-/work/$USER/upwash/runs/$RUN}

if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/work/$USER/.cache/huggingface}
export HF_HUB_OFFLINE=1               # the tokenizer must already be cached (see above)
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://flock-api.cerulean.works}
export WANDB_MODE=${WANDB_MODE:-online}   # offline + `wandb sync` later if nodes cannot reach Flock
export WANDB_DIR=${WANDB_DIR:-/work/$USER/upwash/wandb}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# Single node: bootstrap NCCL over loopback; some nodes' IPoIB addresses are not
# routable even locally (lab5). GPU traffic still goes over NVLink.
export NCCL_SOCKET_IFNAME=${NCCL_IFNAME:-lo}
mkdir -p "$WANDB_DIR" "$(dirname "$OUT")"

log "run=$RUN config=$CONFIG out=$OUT commit=$(git rev-parse --short HEAD) node=$(hostname)"
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
    log "ERROR: nvidia-smi cannot enumerate GPUs on $(hostname); resubmit with --exclude=...,$(hostname)"
    exit 1
fi
NGPU=$(nvidia-smi -L | grep -c '^GPU' || true)
if [[ "$NGPU" -ne "${SLURM_GPUS_ON_NODE:-$NGPU}" ]]; then
    log "ERROR: $NGPU of ${SLURM_GPUS_ON_NODE} GPUs visible on $(hostname); resubmit with --exclude=...,$(hostname)"
    exit 1
fi

.venv/bin/torchrun --standalone --nproc_per_node "$NGPU" -m upwash.train \
    --config "$CONFIG" --name "$RUN" --output-dir "$OUT"

log "done: $OUT"
