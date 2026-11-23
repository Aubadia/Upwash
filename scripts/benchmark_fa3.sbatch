#!/usr/bin/env bash
#SBATCH --job-name=upwash-fa3-bench
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:02:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Upwash repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
.venv/bin/python - <<'PY'
from collections import defaultdict
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
import transformers
from transformers import modeling_flash_attention_utils as flash_utils

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_trainer
from upwash.components.kernels.fa3 import register_upwash_fa3_wrapper
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.config import DataConfig, ModelConfig, OptimizerConfig, RunConfig, ScheduleConfig, TrainingConfig
from upwash.train import train

reference, wrapper = "flash_attention_3", "upwash_fa3"
torch.set_num_threads(2)
torch.manual_seed(17)
print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} transformers={transformers.__version__}")

def model_config(vocab_size, n_layers, d_model, d_ff, n_heads, n_kv_heads, seq_len):
    return ModelConfig(
        vocab_size=vocab_size, n_layers=n_layers, d_model=d_model, d_ff=d_ff,
        n_heads=n_heads, n_kv_heads=n_kv_heads, max_seq_len=seq_len,
        rope_theta=500_000.0, norm_eps=1e-5, embedding_std=0.1, output_std_coeff=1.0,
    )

training = TrainingConfig(
    optimizer=OptimizerConfig(
        lr=0.002, muon_beta=0.9, adam_beta1=0.9, adam_beta2=0.99,
        adam_lr_multiplier=1.0, weight_decay=0.04, fused_adamw=True,
    ),
    precision="bf16", max_grad_norm=1.0,
)

small = model_config(512, 2, 256, 768, 4, 2, 256)
initial_q = build_llama_model(small).model.layers[0].self_attn.q_proj.weight.detach().clone()
updates = {}
with TemporaryDirectory() as tmp:
    tokens = np.random.default_rng(17).integers(0, small.vocab_size, 513, dtype=np.int32)
    np.save(Path(tmp) / "tokens.npy", tokens)
    for backend in (reference, wrapper):
        result = train(RunConfig(
            model=small,
            train_data=DataConfig(tmp, "tokens.npy", "npy", 256, 512),
            training=training,
            schedule=ScheduleConfig(0.0, 0.0, "power", 1.0),
            batch_size=2, device="cuda", seed=17, attention_implementation=backend,
        ))
        assert result.state.global_step == 1
        assert result.model.config._attn_implementation == backend
        q = result.model.model.layers[0].self_attn.q_proj.weight
        assert q.grad is not None and torch.isfinite(q.grad).all()
        update = q.detach().cpu() - initial_q
        assert torch.isfinite(update).all() and torch.count_nonzero(update)
        updates[backend] = update
        print(f"SMOKE {backend} steps=1 q_update_norm={update.norm().item():.8f}")
        del q, result

torch.testing.assert_close(updates[wrapper], updates[reference], rtol=0, atol=1e-6)
native = flash_utils._flash_fn
assert native.__module__.endswith("flash_attn_interface"), native
print(f"SMOKE max_q_update_diff={(updates[wrapper] - updates[reference]).abs().max().item():.8g} native={native.__module__}")

seq_len, batch_size = 1024, 4
config = model_config(128256, 6, 768, 3072, 12, 3, seq_len)
print(f"BENCH params={config.num_parameters()} batch={batch_size} seq={seq_len} compile=off")
tokens = torch.randint(config.vocab_size, (8, batch_size, seq_len + 1), device="cuda")
batches = [(batch[:, :-1].contiguous(), batch[:, 1:].contiguous()) for batch in tokens]
register_upwash_fa3_wrapper()

def measure(backend):
    torch.manual_seed(17)
    model = build_llama_model(config).to("cuda")
    model.set_attn_implementation(backend)
    trainer = build_trainer(
        config=training, model=model, loss_fn=causal_lm_loss,
        optimizer_factory=partial(build_muon_adamw, model_config=config),
    )
    trainer.train(batches[:4])
    torch.cuda.synchronize()
    start = perf_counter()

    def timed_batches():
        while True:
            yield from batches
            torch.cuda.synchronize()
            if perf_counter() - start >= 17.0:
                return

    state = trainer.train(timed_batches())
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    rate = state.global_step * batch_size * seq_len / elapsed
    print(f"RATE {backend} steps={state.global_step} seconds={elapsed:.3f} tokens/s={rate:.1f}")
    return rate

rates = defaultdict(list)
for backend in (reference, wrapper, wrapper, reference):
    rates[backend].append(measure(backend))
mean_ref = sum(rates[reference]) / len(rates[reference])
mean_wrap = sum(rates[wrapper]) / len(rates[wrapper])
print(f"RESULT reference={mean_ref:.1f} wrapper={mean_wrap:.1f} delta_pct={(mean_wrap / mean_ref - 1) * 100:+.2f}")
PY
