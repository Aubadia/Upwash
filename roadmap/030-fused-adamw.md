# 030 — Fused AdamW

Proposed for `feat/acceleration` after model and objective compilation. The
experiment at `8718eb1` measured `2,011,315 tokens/s` with fused AdamW versus
`1,973,828 tokens/s` for its warm baseline in the 150M, eight-GPU ablation.
See [the experiment results](https://github.com/Aubadia/Upwash/blob/8718eb1/experiments/speed150m/RESULTS.md).

## Agreed interface

`OptimizerConfig.fused_adamw` defaults to `False`. With the option enabled,
the existing optimizer factory creates its AdamW group with `fused=True`.
Disabled configurations retain the factory's current AdamW dispatch. Muon
parameter assignment and settings retain their existing behavior.

## CPU reference smoke — 2026-11-20

Ran stock `torch.optim.AdamW` on PyTorch `2.15.0+cu132` using independent FP32
parameter vectors initialized to `[0.25, -0.5, 1.0]`, learning rate `0.004`,
and the same three nonconstant gradients. The Python stdin command used
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -`.
Exit 0. The scalar reference (`fused=False, foreach=False`) and fused path
(`fused=True`) matched parameters and state within `rtol=1e-6, atol=1e-7`.
The maximum observed parameter difference was zero. The reference smoke
establishes CPU execution for this fixture. GPU speed on the user's machine
is unmeasured.

## First Red — 2026-11-20

The factory test runs the real PyTorch fused CPU oracle for two updates
before constructing the Upwash config. It checks explicit fused dispatch,
the final parameter, and AdamW state. The existing factory test also checks
that default dispatch retains the same `fused` and `foreach` values as the
stock optimizer. The config snapshot records both flag values.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/components/optimizers tests/unit/test_training_config.py tests/integration/test_muon_adamw_trainer.py
```

Exit 1, **2 failed, 52 passed in 4.35s**. The reference updates succeed. The
factory case stops at `TypeError` for the unexpected `fused_adamw` keyword
before Upwash runs. The config snapshot also lacks the default field. The
default factory and existing training integrations pass.

Next, the user adds `OptimizerConfig.fused_adamw=False` and passes
`fused=True` to AdamW only when enabled. With the option off, passing
`fused=None` retains PyTorch's automatic dispatch with `foreach=None`.
Rerun the regression to verify Green.

## Verification after adding the field — 2026-11-20

The same regression command exits 1 with **48 failed, 6 passed in 4.28s**.
`fused_adamw` exists but has no default, so existing calls raise a missing
argument `TypeError` before reaching the optimizer. An independent CPU
probe passed `fused_adamw=True` and called the factory with `Linear(4, 3)`.
It reached `Muon.__init__()` and confirmed that Muon rejects the `fused`
argument. The probe used the same CPU environment as the reference with
`PYTHONPATH=src`; exit 0 after checking the expected exception.

Next, default the field to `False`, remove the argument from Muon, and pass
`fused=True` when enabled or `None` when disabled to AdamW. Rerun the
regression to verify both default dispatch and fused numerical behavior.

## Verification after moving the flag to AdamW — 2026-11-20

The user defaulted `fused_adamw` to `False` and moved the argument to AdamW.
The same CPU regression command exits 1 with **1 failed, 53 passed in 4.11s**.
The enabled case matches the fused PyTorch reference for parameters and
optimizer state. The only failure is the default factory test: expected
`fused=None`, observed `fused=False`. In the installed PyTorch 2.15.0 Adam
implementation, automatic foreach selection runs only when both `fused`
and `foreach` are `None`.

Next, change the disabled argument value to `None` and rerun the regression.

## Green — 2026-11-20

The user changed the disabled factory argument to `None`. Rerunning the
targeted command above exits 0 with **54 passed in 4.97s**. The default
factory matches stock AdamW's dispatch settings, while the opt-in factory
matches the fused reference's updates and optimizer state.

A CPU smoke used the real `train()` entry point and the same seeded tiny
Llama, dataset, and configuration twice, toggling only `fused_adamw`. Both
runs completed three updates and the model state dictionaries agreed within
`rtol=1e-6, atol=1e-7`. The Python stdin command used
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
and exited 0.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short
```

Full CPU suite with local loopback access: exit 0, **406 passed in 38.44s**.
GPU execution and throughput remain unmeasured here.
