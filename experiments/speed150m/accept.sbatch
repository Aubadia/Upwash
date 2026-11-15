#!/usr/bin/env bash
#SBATCH --job-name=speed150m-accept
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
export RESULTS
export ACCEPT_WORLD=${ACCEPT_WORLD:-8}
# The exact winning flags are supplied after the controlled refinements.
read -r -a OPTIONS <<< "${ACCEPT_ARGS:---scope objective --fused-adam --prefetch}"
echo "ACCEPT ${OPTIONS[*]}"
NUMERIC=()
for option in "${OPTIONS[@]}"; do
    if [[ "$option" != --prefetch ]]; then NUMERIC+=("$option"); fi
done
bash experiments/speed150m/entry_rank.sh experiments/speed150m/validate_full.py \
    --output "$RESULTS/full-numerics.json" "${NUMERIC[@]}" > "$RESULTS/full-numerics.log" 2>&1
.venv/bin/python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['RESULTS'])
c=json.loads(Path('configs/base150m/base150m-30b-seed2-am1.6.json').read_text())
c['batch_size']=32//int(os.environ['ACCEPT_WORLD'])
c['train_data']['num_tokens']=400*32*4096
c['training']['eval_steps']=100
c['training']['logging_steps']=20
c['schedule']['warmup_ratio']=.05
c['schedule']['decay_ratio']=.2
c['logging']=None
(p/'accept-config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
# Warm exactly the shared train/dev pages before either timed integration run.
# Otherwise the first variant pays cold shared-filesystem faults and the second
# benefits from those reads, as observed in the screening baseline.
.venv/bin/python - <<'PY'
import json,os,time
from pathlib import Path
from upwash.config import load_run_config
from upwash.builder import build_dataset
from upwash.core.data import DataLoader
p=Path(os.environ['RESULTS'])
c=load_run_config(p/'accept-config.json')
started=time.perf_counter()
for data in (c.train_data,c.eval_data):
    for x,y in DataLoader(build_dataset(data),batch_size=32):
        pass
(p/'data-prewarm.json').write_text(json.dumps({'seconds':time.perf_counter()-started})+'\n')
PY
for name in baseline optimized; do
    FLAG=()
    if [[ "$name" == optimized ]]; then FLAG=("${OPTIONS[@]}"); fi
    .venv/bin/torchrun --standalone --nproc_per_node="$ACCEPT_WORLD" --no-python /bin/bash \
        experiments/speed150m/entry_rank.sh experiments/speed150m/train_fast.py \
        --config "$RESULTS/accept-config.json" --result "$RESULTS/$name.json" \
        --output-dir "$RESULTS/$name-model" "${FLAG[@]}" > "$RESULTS/$name.log" 2>&1
done
echo "ACCEPTED $RESULTS"
