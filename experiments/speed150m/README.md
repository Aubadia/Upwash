# 150M training speed experiments

## Using the measured recipe

`recommended.json` selects the verified eight-GPU default-mode composition:
compile model and native CE together, fused AdamW, pinned background prefetch,
and compiled optimizer updates only during the constant-LR plateau. The
`train.sbatch` launcher applies the plan while retaining global batch 32 and the
chosen run's optimizer/schedule/data settings. It logs online to the configured
Flock project. Launch explicitly after choosing the desired training config:

```bash
CONFIG=configs/base150m/smoke.json sbatch experiments/speed150m/train.sbatch
```

See [RESULTS.md](RESULTS.md) for the completed measurements: the recommendation
reaches 2,561,742 tokens/s, 29.8% above the matched warm baseline. The validated
`max-throughput.json` alternative reaches 2,601,377 tokens/s but has substantially
higher compilation cost; select it explicitly with `PLAN` if appropriate:

```bash
PLAN=experiments/speed150m/max-throughput.json CONFIG=configs/base150m/smoke.json sbatch experiments/speed150m/train.sbatch
```

The launcher uses `8gpus`, not dev. It never launches itself from a benchmark.
The 30B instability in task 028 remains unresolved; throughput optimization
does not make the old hyperparameters stable. No formal 30B run was submitted.

The recommended recipe passed full-150M numerical acceptance and actual
eight-GPU Trainer training, WSD warmup/decay, four full dev evaluations, and
serialized HF model reload/logit checks (job 778782). Its validation schedule
is deliberately short and is not evidence of long-run convergence.

Authorized on 2026-11-15: measure fused output/loss, fused AdamW, compilation,
prefetch, and GPU/local-batch configurations, then integrate measured winners.
All GPU execution uses Slurm. Existing training jobs are not modified.

Reference: current Upwash model/optimizer, torch 2.15.0, Transformers 5.22.2,
Triton 3.9.0, FA3 3.0.1. Seed 2, residual scaling 1/6, BF16, context 4096,
global batch 32, Muon LR .004, Adam multiplier 1.6, clipping 1.
Short timing runs do not establish long-run convergence or fix task 028.

Measurements exclude compilation/warmup and use synchronized wall time over
multiple windows, maximum elapsed time across ranks, global tokens/s, and peak
allocated/reserved CUDA memory. Data fetch, transfer, forward, loss, backward,
clipping and optimizer are included. Compiler cold start is reported separately.
Changing GPU count keeps each global batch's sample set and order fixed.
The steady-state benchmark uses constant LR, matching the formal run's plateau.
Scheduled LR and evaluation/export require separate integration acceptance.

Numerical gates run the real unfused reference first, with independent model and
optimizer state. Compare loss (relative 1e-3), gradients (relative L2 .03), and
three-update parameter deltas (relative L2 .10, matching the established BF16
Muon gate). Record actual errors; do not loosen thresholds after failure.
All gradients/parameters must be finite. Shared embedding gradients and output
scaling must be included. Tiny gates precede full-size throughput measurements.

External kernel dependencies require separate user authorization and exact
version pinning. Completed measurements and acceptance evidence are in RESULTS.md.

## Entry points

- `bench.sbatch`: single-variable screening, numerical checks, baseline repeat.
- `refine.sbatch`: longer combination trials and 4/2/1-GPU comparisons.
- `eight.sbatch`: matched eight-GPU comparisons in partition `8gpus`. Its QOS
  permits eight-GPU benchmarking while another job uses dev; dev's QOS has a
  separate eight-GPU/account and two-running-jobs/user limit.
- `liger.sbatch`: explicit eager-loss workaround for the observed Inductor
  `aten.addmm.dtype` lowering failure, with matched native controls.
- `accept.sbatch`: full-150M numerical comparison followed by baseline and
  optimized actual-Trainer runs, WSD warmup/decay, full dev evaluations, HF
  serialization and reloaded-logit comparison. Validation uses 400 updates,
  warmup .05 and decay .2 to exercise all schedule phases; it is not a 30B run.
  It warms the shared train/dev file pages before both variants to avoid the
  cold-data bias observed in the initial screening baselines.
- `train_fast.py`: explicit opt-in composition of the winning flags with the
  existing Trainer. It defaults to local MemoryLogger; `--online` uses the
  configured Flock logger. This does not change normal Upwash defaults.
- `summarize.py DIRECTORY`: report completed timing JSON files only.

Liger 0.8.5 was authorized and installed using `--no-deps --target` under
`/work/asciibase64/upwash/speed150m-deps/liger-0.8.5`. The job's PYTHONPATH adds
this directory; the project's environment and dependency declarations are
unchanged. Fused loss uses FP32 weight-gradient accumulation and scales hidden
states by the existing output multiplier. BF16 operation ordering changes and
must pass the declared numerical gate; no gradient approximation/filter is used.

The objective adapter and plateau-only compiled updates have actual-Trainer CPU
regressions in `tests/integration/test_speed_objective.py`. Both pass; the full
CPU suite passes all 387 tests. GPU throughput and full-model acceptance are
recorded in RESULTS.md and task 029.
