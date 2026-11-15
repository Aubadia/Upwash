#!/usr/bin/env bash
#SBATCH --job-name=speed150m-refine
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
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
    local name=$1 world=$2
    shift 2
    echo "CASE $name WORLD $world $(date -Is)"
    if timeout 900 .venv/bin/torchrun --standalone --nproc_per_node="$world" --no-python \
        /bin/bash experiments/speed150m/rank.sh --name "$name" --output "$RESULTS/$name.json" "$@" \
        > "$RESULTS/$name.log" 2>&1; then echo "PASS $name"; else
        echo "FAIL $name status=$?"; tail -n 15 "$RESULTS/$name.log"
    fi
}
run numeric-native-combined 1 --numerics --scope objective --fused-adam
run native-combined 4 --scope objective --fused-adam --prefetch --warmup 40 --steps 100
run native-no-prefetch 4 --scope objective --fused-adam --warmup 40 --steps 100
run native-prefetch-only 4 --scope objective --prefetch --warmup 40 --steps 100
run numeric-compiled-optimizer 1 --numerics --scope objective --fused-adam --compile-optimizer
run native-compiled-optimizer 4 --scope objective --fused-adam --prefetch --compile-optimizer --warmup 40 --steps 100
run native-reduce-overhead 4 --scope objective --fused-adam --prefetch --mode reduce-overhead --warmup 40 --steps 100
run native-max-autotune 4 --scope objective --fused-adam --prefetch --mode max-autotune --warmup 40 --steps 100
run native-2gpu 2 --scope objective --fused-adam --prefetch --warmup 40 --steps 100
run native-1gpu 1 --scope objective --fused-adam --prefetch --warmup 40 --steps 100
run baseline-warm 4 --warmup 40 --steps 100
run native-repeat 4 --scope objective --fused-adam --prefetch --warmup 40 --steps 100
echo "FINISHED $RESULTS"
