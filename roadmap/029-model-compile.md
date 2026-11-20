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

The proposal is pending agreement. The integration test still uses its
existing absolute tolerance and remains Red.

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

## Agreed follow-up — Compile model and loss together

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
