#!/usr/bin/env bash
#SBATCH --job-name=300m-group-accept
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=/work/asciibase64/upwash/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD:$PWD/src"
export HF_HUB_OFFLINE=1 HF_HOME=/work/asciibase64/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_SOCKET_IFNAME=lo WANDB_MODE=disabled
export RESULTS="/work/asciibase64/upwash/validation/group300m-${SLURM_JOB_ID}"
mkdir -p "$RESULTS"
for label in 0.002 0.0025; do
    MANIFEST="experiments/sweep300m/300m-60b-base-lr${label}-power-p1.json"
    bash experiments/speed150m/entry_rank.sh experiments/sweep300m/validate_grouped.py \
        --manifest "$MANIFEST" --output "$RESULTS/numerics-${label}.json" \
        > "$RESULTS/numerics-${label}.log" 2>&1
    .venv/bin/torchrun --standalone --nproc_per_node=8 --no-python /bin/bash \
        experiments/speed150m/entry_rank.sh experiments/sweep300m/train_grouped.py \
        --manifest "$MANIFEST" --smoke-steps 400 --output-dir "$RESULTS/model-${label}" \
        --result "$RESULTS/integration-${label}.json" --scope objective --mode default \
        --fused-adam --prefetch --compile-optimizer > "$RESULTS/integration-${label}.log" 2>&1
done
.venv/bin/python experiments/sweep300m/check_acceptance.py "$RESULTS"
