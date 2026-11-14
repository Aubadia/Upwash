#!/usr/bin/env bash
#SBATCH --job-name=prep-pool
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus             # CPU partitions reject ACD115198; GPUs stay idle
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                  # Nano4 grants 12 cores per GPU: 8 GPUs -> 96 cores
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00               # ~2 h estimated; reruns skip finished shards
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

# Usage, from the repository root:
#   TARGET=205e9 sbatch scripts/prepare_pretrain.sbatch
# Interrupted or timed out? Submit again with the same TARGET/SEED: finished
# shards are reused. HF_HOME stays at its default (~/.cache/huggingface) so the
# saved HF token authorizes the gated Llama tokenizer download.

cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
TARGET=${TARGET:-205e9}
SEED=${SEED:-0}
OUT=${OUT:-/work/$USER/fineweb-edu-llama3-upwash}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-96}}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# ~96 processes share the per-user thread limit (ulimit -u, 2048 on the login
# node). hf_xet starts a large thread pool per process (job 773185 died with
# "OS can't spawn worker thread"), so download over plain HTTP and keep
# per-process thread pools small.
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 ARROW_IO_THREADS=2

echo "[$(date +%T)] ulimit -u=$(ulimit -u) node=$(hostname) target=$TARGET seed=$SEED workers=$WORKERS out=$OUT commit=$(git rev-parse --short HEAD)"
# The pool streams from the Hugging Face Hub; stop early if this node is offline.
if ! timeout 20 curl -sfI https://huggingface.co >/dev/null; then
    echo "ERROR: $(hostname) cannot reach huggingface.co; run on the login node or ask the TA" >&2
    exit 1
fi

.venv/bin/python scripts/prepare_pretrain.py \
    --output-dir "$OUT" --target-tokens "$TARGET" --seed "$SEED" --workers "$WORKERS"
echo "[$(date +%T)] done: $OUT/manifest.json"
