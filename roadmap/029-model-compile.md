# 029 — Opt-in model compilation

Branch `feat/acceleration`. The model-only entry-point behavior follows
`origin/experiment` at `8718eb1`.

## Agreed behavior

- `RunConfig` accepts optional `compile_model=False` and
  `compile_mode="default"`. The default training path remains eager.
- When enabled, `train(config)` compiles the model with the selected mode
  and supplies the compiled callable to Trainer. `TrainResult.model` still
  exposes the original model. The first test uses the single-process path.

## First Red — 2026-11-20

The CPU unit test uses a compiler double that returns a distinct wrapper. It
checks that the wrapper reaches Trainer, that default and explicit modes are
forwarded, and that the eager path never calls the compiler. The Trainer double
isolates entry-point wiring. Actual compiler execution, numerical agreement,
and distributed compilation with the option enabled remain unverified.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/test_llama_init.py tests/unit/test_adaptor.py tests/unit/test_model_config.py
```

Exit 1, **2 failed, 92 passed in 4.51s**. The default-mode and explicit-mode
cases stop at `TypeError: RunConfig.__init__() got an unexpected keyword
argument 'compile_model'`. The eager case and existing regressions pass.

Next, the user implements the `RunConfig` fields and compile selection in
`train()`, then rerun the regression to verify Green.

## Green — 2026-11-20

The user added the config fields and calls `model.compile(mode=...)` in place
before optional DDP wrapping. Trainer uses the compiled forward and the result
retains the original `nn.Module`. The first rerun gave **2 failed, 92 passed**
because the test assumed a distinct wrapper. The revised compiler double
changes the forward output, allowing the test to verify which forward Trainer
uses while retaining assertions for the chosen mode and returned model.
Rerunning the command above gives exit 0, **94 passed in 4.64s**.

An additional CPU smoke ran the real `train()` path with a tiny Llama, default
Inductor mode, five packed sequences, and one update per run. Both runs used
the same seed. The command exited 0 and the resulting state dictionaries
agreed within `rtol=1e-4, atol=1e-5`.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import torch
from upwash.train import train

make_run_config = run_path('tests/unit/test_train.py')['make_run_config']
make_optimizer_config = run_path('tests/conftest.py')['make_optimizer_config'].__wrapped__()
with TemporaryDirectory(prefix='upwash-compile-smoke-') as directory, torch.random.fork_rng(devices=[]):
    root = Path(directory)
    eager = train(make_run_config(root, make_optimizer_config, batch_size=5))
    compiled = train(make_run_config(root, make_optimizer_config, batch_size=5, compile_model=True))
    assert eager.state.global_step == compiled.state.global_step == 1
    torch.testing.assert_close(compiled.model.state_dict(), eager.model.state_dict(), rtol=1e-4, atol=1e-5)
print('CPU Inductor smoke passed: one tiny Llama update agrees with eager')
PY
```

## Two-rank CPU verification — 2026-11-20

Extended the existing DDP integration with two compiled Gloo ranks versus a
single eager process. Both paths train the same tiny Llama for four updates
with a global batch of four, Muon + AdamW, WSD, and FP32 forward. The compiled
case uses default Inductor and disables clipping. The eager clipping and
no-clipping cases retain their existing inputs and checks.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/integration/test_ddp_training.py tests/unit/test_train.py tests/unit/test_train_ddp.py
```

Exit 1, **1 failed, 18 passed in 44.31s**, with local loopback socket access.
Both compiled ranks reached four updates and tore down their process groups.
The `compiled-model` case first exceeded the existing absolute tolerance of
`5e-4` at `model.layers.0.mlp.gate_proj.weight`. Both eager DDP cases passed.

A four-way CPU diagnostic reused the same tiny model, seed, batches, optimizer,
and schedule. Every pair of DDP replicas was bit-identical after four updates.
All four comparisons had their largest absolute difference in `gate_proj.weight`.

| Comparison | Max absolute parameter difference |
| --- | ---: |
| Eager DDP vs eager single process | `1.32576e-4` |
| Compiled single process vs eager single process | `6.40780e-4` |
| Compiled DDP vs eager DDP | `6.62878e-4` |
| Compiled DDP vs eager single process | `6.40780e-4` |

The gap reproduces with one process. A separate fixed-LR, four-update CPU
diagnostic compared plain SGD and Muon + AdamW on the same batches. The
first `gate_proj.weight` gradient differed by `1.76951e-8` in both cases. The
final weight differed by `2.98023e-8` under SGD and `7.51197e-4` under Muon +
AdamW. The optimizer builder routes this matrix to Muon. This is consistent
with sensitivity to small gradient changes and does not establish which Muon
operation accounts for the gap.

Next, agree on the compiled-run numerical acceptance criterion before
changing the integration test's tolerance. The regression remains Red at
the inherited threshold. GPU execution is still unverified.

## Proposed numerical acceptance — 2026-11-20

For this four-update CPU Inductor + Muon smoke, measure
`norm(compiled - eager) / norm(eager - initial)` in L2, both across all
parameters and per tensor. Count tied parameters once and retain the existing
checks for actual training progress, exact replica agreement, and teardown.
Keep the eager DDP tolerance unchanged. Proposed limits for the compiled
case are `1%` globally and `2%` per tensor. These apply to this short CPU
scenario. The experiment's 150M BF16 acceptance uses the same aggregate
update metric on a different workload and has its own tolerance.

The diagnostic reuses the DDP fixture and accumulates squared errors and
update norms in FP64. The first diagnostic run reached the comparisons but
could not match the extra tied `lm_head.weight` state-dict key against the
DDP parameter mapping. The correction uses `named_parameters()` in all runs.

| Comparison, token seed 0 | Global update error | Highest per-tensor error |
| --- | ---: | ---: |
| Eager DDP vs eager single | 0.025912% | 0.204386% |
| Compiled single vs eager single | 0.192352% | 1.023451% |
| Compiled DDP vs eager DDP | 0.191218% | 1.010094% |
| Compiled DDP vs eager single | 0.192353% | 1.023451% |

The highest per-tensor error in each seed-0 pair came from `gate_proj.weight`.
With token seed 1 and the same model seed, compiled DDP versus eager single
measured `0.000036%` globally and `0.000088%` at the worst tensor. Each
diagnostic ran locally with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`,
`TORCHINDUCTOR_COMPILE_THREADS=1`, and `PYTHONPATH=src:tests/integration` via
`.venv/bin/python` on a temporary script. Both corrected runs exited 0.

At this point the integration test still used its existing absolute tolerance
and remained Red. The approved criterion appears below.

## Muon orthogonalization precision diagnostic — 2026-11-20

The installed PyTorch 2.15.0 Muon casts its Newton–Schulz input to BF16 and
exposes no precision argument for that operation. Reused the failing token
seed and trained four single-process updates using the production WSD schedule
and optimizer wiring. For each precision, compared eager and compiled runs
using the same precision on both sides. The FP32 case replaced only the cast
in a process-local copy of the PyTorch orthogonalization function.

| Orthogonalization dtype | Maximum absolute parameter gap | Worst per-tensor relative update gap |
| --- | ---: | ---: |
| BF16 | `6.40780e-4` | `1.02345150%` |
| FP32 | `1.01328e-6` | `0.00023740%` |

The diagnostic ran via `.venv/bin/python` with `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=2`, `TORCHINDUCTOR_COMPILE_THREADS=1`, and
`PYTHONPATH=src:tests/integration`. Exit 0. The result identifies a way to
reduce eager/compiled disagreement on this workload. FP32 optimizer behavior,
GPU throughput, and convergence are outside this diagnostic. Repository
production code and the failing integration assertion were left as-is.

## Initial objective-scope design — Superseded by the single switch below

`RunConfig.compile_scope="model"` preserves current model-only
behavior when compilation is enabled. With `compile_scope="objective"`,
compile one callable from `(inputs, targets)` to the scalar causal LM loss.
Both training and evaluation already pass through `Trainer.compute_loss`.
The optimizer, scheduler, and step loop are outside this proposal.

Before proposing the public API, ran the experiment's native `Objective` class
from `8718eb1:experiments/speed150m/bench.py` in a standalone CPU smoke. Loaded
only that class, using the local Llama adapter and causal LM loss. Compared
eager and default-mode Inductor from independent models with the same seed and
a fixed two-by-eight token batch. The Python stdin command used
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`.
Exit 0. Absolute loss gap was `2.38419e-7` and maximum absolute gradient gap
was `1.49012e-7`. This verifies the reference's CPU behavior. Upwash's objective
scope entry-point test follows below. GPU speed remains unmeasured here.

### First objective-scope Red — 2026-11-20

The entry-point test uses a compiler double to observe one compiled call
containing both the model forward and cross-entropy. The normal Trainer and
optimizer run one batch with real gradients. The test also asserts that the
result returns the original model and that its parameters actually change.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/test_llama_init.py tests/unit/test_adaptor.py tests/unit/test_model_config.py
```

Exit 1, **1 failed, 94 passed in 4.86s**. The new case stops at
`TypeError: RunConfig.__init__() got an unexpected keyword argument 'compile_scope'`.
Its execution and update assertions have yet to run. The earlier compiled DDP
numerical Red was not rerun. Next, the user adds the config field and routes
the objective callable, then rerun this regression to verify Green.

## Agreed single compile switch — 2026-11-20

`RunConfig.compile_scope` alone selects eager (`"none"`, the default), compiled
model forward (`"model"`), or compiled forward and loss (`"objective"`).
`compile_mode` keeps selecting the PyTorch mode. Remove `compile_model` from
the config and entry-point dispatch. Trainer results still return the original
model.

The CPU tests now exercise the three scopes, the default scope, and mode
forwarding. The DDP case passes its scope to the constructor and verifies that
each model-scope rank invokes the real compiler before comparing parameters.
This prevents accidental eager execution from satisfying the compiled case.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/test_llama_init.py tests/unit/test_adaptor.py tests/unit/test_model_config.py tests/integration/test_ddp_training.py
```

Exit 1, **5 failed, 95 passed in 25.69s**, with local loopback socket access.
The user has added `compile_scope` with default `"objective"`, while `train()`
still checks `compile_model`. One failure is the config default, two are the
model modes that went eager, one is the objective that went eager, and one is
the DDP case where neither rank invoked the compiler. Both eager DDP cases
passed. The compiled DDP numerical assertion has not been reached with the
new API, and its prior tolerance issue remains open.

Next, the user makes `compile_scope` the sole switch with default `"none"`
and routes the model and objective scopes. Rerun the command to verify wiring
and reach the compiled DDP numerical assertion.

### Verification after adding the objective wrapper — 2026-11-20

The user removed the old boolean and added scope dispatch and an objective
wrapper. Running the same unit regression command from the first objective
Red gives exit 1, **11 failed, 86 passed in 5.94s**. Explicit eager and both
model-mode cases pass. The default remains `"objective"`, so default callers
also enable the wrapper. `Trainer.compute_loss()` still calls `model(inputs)`;
the wrapper requires `(inputs, targets)`, so the objective path raises
`TypeError` before producing a loss. Source inspection also shows that the
objective branch overwrites `model`, which `TrainResult` returns.

Next, change the scope default to `"none"`, keep the wrapper in `trained`, and
route both inputs and targets through `compute_loss()` for the objective.
Real two-rank tests were not rerun for this check.

### Verification after restoring default eager — 2026-11-20

The user set the default scope to `"none"` and retained the original model
separately from the objective wrapper. The first objective Red unit command
now gives **1 failed, 96 passed in 4.48s**, exit 1. Only the objective case
fails. `Trainer.compute_loss()` still calls the wrapper with inputs alone,
raising `TypeError: _CausalLMObject.forward() missing 1 required positional
argument: 'targets'`. The entry point still passes `causal_lm_loss` to Trainer.
Source inspection also shows that DDP wraps the original model instead of the
selected training module. Real two-rank tests were not rerun.

Next, have the objective path pass `loss_fn=None` and use `model(inputs,
targets)` for that case in `compute_loss()`. Have DDP wrap `trained` so the
selected objective path also works across ranks.

The user subsequently changed DDP to wrap `trained`. The focused command
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py -k compiles_model_and_loss`
still exits 1 with **1 failed, 12 deselected in 8.07s** for missing `targets`.
The remaining loss dispatch is at the `build_trainer()` call and at the start
of `Trainer.compute_loss()`. Multi-rank execution was not rerun.

### Objective scope Green — 2026-11-20

The user wired `loss_fn=None` for the objective path. `Trainer.compute_loss()`
now passes inputs and targets to that wrapper. The unit command from the
first objective Red exits 0 with **97 passed in 4.62s**.

The single-process FP32 smoke ran the real `train()` entry point with scopes
`"none"` and `"objective"`, each on the same seeded tiny Llama. Each run used
one batch of five packed sequences, one optimizer update, and a scheduled
evaluation on eight held-out tokens. Both returned an HF Llama. The resulting
parameters agree within `rtol=1e-4, atol=1e-5`, and the reported eval loss is
identical. The Python stdin command used
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
and exited 0. No GPU run or objective-scope two-rank run was performed.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short
```

The full CPU suite with local loopback access exits 1 with **403 passed,
1 failed in 31.03s**. The single-switch model-scope DDP case reaches the same
numerical assertion as before. `gate_proj.weight` differs at 2 of 512 elements
above the existing `5e-4` absolute tolerance, with maximum `6.40780e-4`.
The relative-update criterion was pending agreement at this point.

### Agreed compiled DDP acceptance — 2026-11-20

The user approved relative L2 update error of at most `1%` globally and `2%`
per parameter tensor for the four-update CPU compiled-model case. Each error
is normalized by the eager update from the shared initial parameters. The
eager DDP absolute tolerance remains `5e-4`. Both DDP paths still require
bit-identical replicas, four updates, and process-group teardown.

Reran the full CPU suite using the command above with local loopback access.
Exit 0, **404 passed in 29.43s**. This verifies model-scope DDP under the
approved limits and the existing eager cases. Objective-scope multi-rank
execution and GPU throughput remain unverified.

### Objective-scope DDP Green — 2026-11-20

Added `compiled-objective` to the existing two-rank Gloo integration. The test
runs real default-mode CPU Inductor, compares four updates with single-process
eager using the same L2 limits as compiled-model, and checks that both ranks
invoke compilation, finish with bit-identical replicas, and destroy their
process groups. The two eager cases and the model-scope case still pass.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short
```

Exit 0, **405 passed in 41.09s** with local loopback access. Objective-scope
DDP CPU updates are verified. GPU execution and speed are unmeasured here.

## Compile the DDP wrapper — 2026-11-21

For multiple ranks, the training entry point builds the model or causal LM
objective chosen by `compile_scope`, wraps it in DDP, then passes the wrapper
to `torch.compile(..., mode=config.compile_mode)`. This follows the ordering
in [the experiment's 150M training entry point](https://github.com/Aubadia/Upwash/blob/8718eb1/experiments/speed150m/train_fast.py#L69-L81).
Single-process runs use the selected module's `.compile()` method.

```sh
PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train_ddp.py -k compiles_after_ddp
```

Before the production change, this command exited 1 with **3 failed,
6 deselected**. Each case observed compilation of the inner model before
DDP wrapping. The cases now expect device placement, optional attention
selection, DDP wrapping, then compilation of that wrapper.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short
```

Exit 0, **409 passed in 19.98s** on the local Mac with loopback access.
The real two-rank Gloo tests passed for both compile scopes with the existing
update tolerances and bit-identical replicas. CUDA/NCCL is unverified for
this ordering. Next, compare both orderings with a matched 8-GPU Slurm run,
recording compilation time and steady-state throughput.

## Deferred optimizer compilation — 2026-11-20

The experiment at `8718eb1` uses Inductor for `optimizer.step()` when
all WSD group LRs equal their base rates. A standalone probe used the
experiment's `compiled_update.py`, PyTorch `2.15.0+cu132`, independent
`Linear(4, 3)` copies, and the Upwash Muon + AdamW factory. The six WSD
factors were `0.5, 1, 1, 1, 0.5, 0`. All compiled candidates dispatched
to the compiled step on updates 2 through 4.

A strict initial CPU comparison exited 1 on the first compiled update with
a maximum absolute Muon weight difference of `9.12771e-4`. Measuring all
six updates gave these results. Relative L2 error is normalized by the
eager update from the shared initialization.

| Backend and gradient fixture | Final Muon update gap | Maximum absolute AdamW bias gap |
| --- | ---: | ---: |
| CPU Inductor, rank 2 | 37.9965% | `5.96e-8` |
| RTX 3080 Inductor, rank 2 | 37.9965% | `2.98e-8` |
| CPU `aot_eager`, rank 2 | 0% | `2.98e-8` |
| CPU Inductor, rank 3 and fused AdamW | 0.737474% | `0` |
| RTX 3080 Inductor, rank 3 and fused AdamW | 0.737474% | `0` |

The measurement scripts ran via `.venv/bin/python` with `OMP_NUM_THREADS=2`,
`TORCHINDUCTOR_COMPILE_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, and
`PYTHONPATH=src`. CPU used `CUDA_VISIBLE_DEVICES=''` and the laptop used
`CUDA_VISIBLE_DEVICES=0`. Both measurement runs exited 0 after reporting
their differences and checking dispatch. The user deferred optimizer
compilation and chose [CUDA batch prefetch](032-cuda-batch-prefetch.md) as
task 032.
