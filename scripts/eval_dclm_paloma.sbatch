#!/usr/bin/env bash
#SBATCH --job-name=upwash-dclm-paloma
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/work/asciibase64/upwash/dclm-150m/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
run_root=/work/asciibase64/upwash/dclm-150m
export TMPDIR="$run_root/tmp/$SLURM_JOB_ID"
export HF_HOME="$run_root/cache/huggingface"
export FLOCK_EVAL_HUB="$run_root/cache/eval-hub"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
python="$PWD/.venv/bin/python"
flock_venv="${DCLM_FLOCK_VENV:-/home/asciibase64/flock/.venv}"
purelib='import sysconfig; print(sysconfig.get_paths()["purelib"])'
upwash_site=$("$python" -c "$purelib")
flock_site=$("$flock_venv/bin/python" -c "$purelib")
export PYTHONPATH="$PWD/src:$PWD:$upwash_site:${flock_venv%/*}:$flock_site${PYTHONPATH:+:$PYTHONPATH}"
paloma_pin=65cd6fc59dba021b21db414fa5e8d7765ffbe5e6
if [[ ! -d "$FLOCK_EVAL_HUB/allenai/paloma/$paloma_pin" ]]; then
    "$flock_venv/bin/flock" eval hub pull "cerulean-labs/allenai/paloma@$paloma_pin" --to "$FLOCK_EVAL_HUB"
fi
if [[ " $* " == *" --flock-run-id "* ]]; then
    set -a
    source .env
    set +a
    export WANDB_MODE=online
fi
tokenizer=/work/asciibase64/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
"$python" scripts/eval_dclm_paloma.py --tokenizer-dir "$tokenizer" "$@"
