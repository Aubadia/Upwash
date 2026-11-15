#!/usr/bin/env bash
#SBATCH --job-name=speed150m
#SBATCH --account=ACD115198
#SBATCH --partition=dev
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
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_SOCKET_IFNAME=lo
export WANDB_MODE=disabled
RESULTS="/work/asciibase64/upwash/speed150m/${SLURM_JOB_ID}"
mkdir -p "$RESULTS"
cp experiments/speed150m/bench.py "$RESULTS/bench.py"
cp configs/base150m/base150m-30b-seed2-am1.6.json "$RESULTS/config.json"
run() {
    local name=$1 world=$2
    shift 2
    echo "CASE $name WORLD $world $(date -Is)"
    if timeout 900 .venv/bin/torchrun --standalone --nproc_per_node="$world" --no-python \
        /bin/bash experiments/speed150m/rank.sh --name "$name" --output "$RESULTS/$name.json" "$@" \
        > "$RESULTS/$name.log" 2>&1; then
        echo "PASS $name"
    else
        echo "FAIL $name status=$?"
        tail -n 15 "$RESULTS/$name.log"
    fi
}
WORLD=${BENCH_WORLD:-8}
if [[ "${BENCH_STAGE:-screen}" == screen ]]; then
    run numeric-fused-adam 1 --numerics --fused-adam --mode eager
    run numeric-fused-loss 1 --numerics --fused-loss --mode eager
    run baseline "$WORLD" --profile
    run fused-adam "$WORLD" --fused-adam
    run compiled-loss "$WORLD" --scope objective
    run prefetch "$WORLD" --prefetch
    run reduce-overhead "$WORLD" --mode reduce-overhead
    run max-autotune "$WORLD" --mode max-autotune
    run compiled-optimizer "$WORLD" --compile-optimizer
    if .venv/bin/python -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$RESULTS/numeric-fused-loss.json"; then
        run fused-loss "$WORLD" --fused-loss
        run combined "$WORLD" --fused-loss --fused-adam --prefetch
    fi
    run baseline-repeat "$WORLD"
else
    run "${BENCH_NAME:?}" "$WORLD" ${BENCH_ARGS:-}
fi
echo "FINISHED $RESULTS"
