#!/usr/bin/env bash
#SBATCH --job-name=base150m-fa3-build
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/work/asciibase64/upwash/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
REPO=$PWD
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$REPO/.venv/bin:$PATH"
export UV_CACHE_DIR=/work/asciibase64/upwash/cache/uv
export MAX_JOBS=8 NVCC_THREADS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
FA3=/work/asciibase64/upwash/vendor/flash-attention
[[ "$(git -C "$FA3" rev-parse HEAD)" == f63c472c4dc19237c0635d6a629df065ff64796e ]]
uv pip install --python "$REPO/.venv/bin/python" --no-build-isolation --no-deps "$FA3/hopper"
python - <<'PY'
import torch
import json
from pathlib import Path
from importlib.metadata import version
from flash_attn_interface import flash_attn_func
q=torch.randn(2,128,12,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(2,128,3,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
v=torch.randn_like(k,requires_grad=True)
out=flash_attn_func(q,k,v,causal=True)
out.float().square().mean().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))
record={'torch':torch.__version__,'transformers':version('transformers'),'flash_attn_3':version('flash-attn-3'),'source_commit':'f63c472c4dc19237c0635d6a629df065ff64796e','cuda_toolkit':'13.0','gpu':torch.cuda.get_device_name()}
Path('/work/asciibase64/upwash/fa3-build.json').write_text(json.dumps(record,indent=2)+'\n')
print('FA3_BUILD_AND_BACKWARD_OK',record)
PY
