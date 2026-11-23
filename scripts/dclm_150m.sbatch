#!/usr/bin/env bash
#SBATCH --job-name=upwash-dclm150m
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=/work/asciibase64/upwash/dclm-150m/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
run_root=/work/asciibase64/upwash/dclm-150m
mkdir -p "$run_root"/{cache,runs,logs} "$run_root/tmp/$SLURM_JOB_ID"
export TMPDIR="$run_root/tmp/$SLURM_JOB_ID"
export HF_HOME="$run_root/cache/huggingface"
export DCLM_EXISTING_HF_CACHE="/work/asciibase64/.cache/huggingface/hub"
export TRITON_CACHE_DIR="$run_root/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$run_root/cache/inductor"
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

python="$PWD/.venv/bin/python"
flock_venv="${DCLM_FLOCK_VENV:-/home/asciibase64/flock/.venv}"
# Put Upwash's package versions first. The Flock checkout and its Lance are
# available to both the runner and the credential subprocess.
purelib='import sysconfig; print(sysconfig.get_paths()["purelib"])'
upwash_site=$("$python" -c "$purelib")
flock_site=$("$flock_venv/bin/python" -c "$purelib")
export PYTHONPATH="$PWD/src:$PWD:$upwash_site:${flock_venv%/*}:$flock_site${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == --test ]]; then
    shift
    "$python" -m pytest -q -p no:cacheprovider tests/unit/test_dclm_150m.py "$@"
    exit
fi

if [[ "${1:-}" == --preflight ]]; then
    "$python" scripts/train_dclm_150m.py "$@"
    exit
fi

fa3_build=/work/asciibase64/upwash/vendor/flash-attention/hopper/build/lib.linux-x86_64-cpython-314
if [[ -d "$fa3_build" ]]; then
    export PYTHONPATH="$PYTHONPATH:$fa3_build"
fi
"$python" -m torch.distributed.run --standalone --nproc-per-node=8 scripts/train_dclm_150m.py "$@"
