# 031 — FlashAttention-3 selection

The user chose FA3 before CUDA batch prefetch. The behavioral reference is
the experiment at `8718eb1`. It sets the HF attention backend after placing
the model on the runtime device. Its FA3 build pins `flash-attn-3` to upstream
commit `f63c472c4dc19237c0635d6a629df065ff64796e` and sets
`FLASH_ATTENTION_DISABLE_SM80=TRUE`.

## Agreed interface

`RunConfig.attention_implementation: str | None = None`. `None` preserves
the model's Transformers default. An explicit backend, including
`"flash_attention_3"`, is sent to `model.set_attn_implementation()` after
device placement and before wrapping or compilation. The first test checks
this entry-point behavior with CPU test doubles. CUDA numerical acceptance
requires the FA3 extension.

## Laptop preflight — 2026-11-20

The laptop's RTX 3080 is SM 8.6. The experiment binary excludes SM8x, but the
[pinned upstream setup](https://raw.githubusercontent.com/Dao-AILab/flash-attention/f63c472c4dc19237c0635d6a629df065ff64796e/hopper/setup.py)
can include SM80 forward and backward variants. The installed Transformers
5.22.2 compatibility matrix allows FA3 on CUDA compute-capability major 8
and above. The upstream README highlights Hopper. Actual laptop FA3
execution remains unverified.

The Python stdin preflight used
`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
with local GPU access. A real HF Llama raises `ImportError` when selecting
`flash_attention_3` because the extension is absent. The same model runs SDPA
forward and backward in CUDA BF16 with finite gradients. PyTorch's extension
preflight reports a CUDA major-version mismatch between system `nvcc` 12.8
and the PyTorch `2.15.0+cu132` build. The diagnostic exits 0 after checking
those two expected environment errors. No dependencies were installed.

## First Red — 2026-11-20

The CPU wiring test simulates a two-rank run and records model placement,
backend selection, compilation, and DDP wrapping. The default case expects
the normal Transformers backend selection. The explicit case expects
`flash_attention_3` after placement and before compilation and wrapping.
The setter, compiler, and DDP are test doubles. The tiny model still performs
two real CPU optimizer updates when the path reaches Trainer.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train_ddp.py tests/unit/test_train.py tests/unit/test_adaptor.py tests/unit/test_llama_init.py
```

Exit 1, **1 failed, 41 passed in 8.55s**. The explicit case stops at
`TypeError: RunConfig.__init__() got an unexpected keyword argument 'attention_implementation'`.
The default case and existing regressions pass.

Next, the user adds the field and forwards explicit values to the model
setter. Rerun the regression to verify Green. The laptop CUDA smoke still
requires a compatible FA3 build with SM8x enabled.

## Verification after adding FA3 dispatch — 2026-11-20

The original CPU regression exits 0 with **42 passed in 4.42s**. The FA3
string reaches the setter in the agreed order, and omitting the field leaves
the default path alone. The agreed API also accepts other explicit backends.
Adding an explicit `"eager"` case gives **1 failed, 42 passed in 4.92s** with
the same command. The setter call is missing because `train()` only forwards
the exact `"flash_attention_3"` string.

The real single-GPU `train()` path confirmed the same mismatch. A tiny
Llama completed one BF16 update on the RTX 3080 Laptop after being configured
with `"eager"`, but Transformers still reported `"sdpa"`. The Python stdin
command used `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
with local GPU access and exited 1 on the explicit backend assertion.

Next, forward every non-`None` explicit backend and rerun the CPU regression
and real CUDA smoke. The FA3 kernel itself was not tested in this round.

## Backend selection Green — 2026-11-20

The user changed the guard to forward every non-`None` backend. The targeted
CPU regression command above exits 0 with **43 passed in 5.71s**. The real
`train()` smoke on the RTX 3080 Laptop completes one BF16 update per backend.
Transformers selects `"sdpa"` with the default `None` and `"eager"` when
explicitly requested. The Python stdin command used
`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
with local GPU access, exit 0.

Actual FA3 forward, backward, and throughput remain unverified. The FA3
extension is absent, and the available local build toolchain still has the
CUDA major-version mismatch recorded above.
