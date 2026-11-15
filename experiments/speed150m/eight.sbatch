#!/usr/bin/env bash
#SBATCH --job-name=speed150m-eight
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
export PYTHONPATH="$PWD/src:/work/asciibase64/upwash/speed150m-deps/liger-0.8.5"
export HF_HUB_OFFLINE=1 HF_HOME=/work/asciibase64/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_SOCKET_IFNAME=lo WANDB_MODE=disabled
RESULTS="/work/asciibase64/upwash/speed150m/${SLURM_JOB_ID}"
mkdir -p "$RESULTS"
cp experiments/speed150m/bench.py "$RESULTS/bench.py"
run() {
    local name=$1
    shift
    echo "CASE $name $(date -Is)"
    if timeout 1200 .venv/bin/torchrun --standalone --nproc_per_node=8 --no-python \
        /bin/bash experiments/speed150m/rank.sh --name "$name" --output "$RESULTS/$name.json" --warmup 40 --steps 100 "$@" \
        > "$RESULTS/$name.log" 2>&1; then echo "PASS $name"; else
        echo "FAIL $name status=$?"; tail -n 15 "$RESULTS/$name.log"
    fi
}
run baseline
run fused-adam --fused-adam
run prefetch --prefetch
run compiled-loss --scope objective
run native-combined --scope objective --fused-adam --prefetch --compile-optimizer
run native-max-combined --scope objective --fused-adam --prefetch --compile-optimizer --mode max-autotune
run liger-fallback --fused-loss --eager-fused-loss
run liger-combined --fused-loss --eager-fused-loss --fused-adam --prefetch --compile-optimizer
run baseline-repeat
run native-repeat --scope objective --fused-adam --prefetch --compile-optimizer
echo "FINISHED $RESULTS"
