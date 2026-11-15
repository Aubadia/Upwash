#!/usr/bin/env bash
#SBATCH --job-name=upwash-fast
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=14:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn020,25a-hgpn039,25a-hgpn056,25a-hgpn062,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn173
#SBATCH --output=/work/asciibase64/upwash/logs/%x-%j.out
# Explicit user launch only. Benchmarks never submit this long training job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${CONFIG:?Set CONFIG to the chosen training run config}"
PLAN=${PLAN:-experiments/speed150m/recommended.json}
RUN=${RUN:-fast-$(basename "$CONFIG" .json)-${SLURM_JOB_ID}}
OUT=${OUT:-/work/asciibase64/upwash/runs/$RUN}
mkdir -p "$OUT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export CONFIG PLAN RUN OUT
export PYTHONPATH="$PWD/src:/work/asciibase64/upwash/speed150m-deps/liger-0.8.5"
export HF_HUB_OFFLINE=1 HF_HOME=/work/asciibase64/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 NCCL_SOCKET_IFNAME=lo
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://flock-api.cerulean.works}
export WANDB_MODE=online WANDB_DIR=/work/asciibase64/upwash/wandb
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text())
assert plan['acceptance_passed'], 'The selected plan has not passed GPU acceptance'
assert int(os.environ['SLURM_GPUS_ON_NODE']) == plan['world_size'], 'Allocation must match the measured plan'
config=json.loads(Path(os.environ['CONFIG']).read_text())
config['batch_size']=plan['global_batch']//plan['world_size']
config['output_dir']=os.environ['OUT']
if config.get('logging'):
    config['logging']['name']=os.environ['RUN']
Path(os.environ['OUT'],'run-config.json').write_text(json.dumps(config,indent=2)+'\n')
Path(os.environ['OUT'],'performance-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
Path(os.environ['OUT'],'flags.txt').write_text('\n'.join(plan['flags'])+'\n')
PY
mapfile -t FLAGS < "$OUT/flags.txt"
exec .venv/bin/torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE}" --no-python \
    /bin/bash experiments/speed150m/entry_rank.sh experiments/speed150m/train_fast.py \
    --config "$OUT/run-config.json" --result "$OUT/result.json" --online "${FLAGS[@]}"
