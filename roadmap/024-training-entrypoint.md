# 024 — Configured Training Entry Point

## Agreed Direction — 2026-11-14

The user approved `RunConfig -> train(config) -> TrainResult(model, state)`.
Production implementation remains owned by the user. Compose existing configs.

First-increment API:

- `upwash.config.RunConfig`: `model`, `train_data`, `eval_data`, `training`,
  `schedule`, `batch_size`, `eval_batch_size`, `device`, `seed`, `logging`.
- `upwash.train.train(config)` returns `upwash.train.TrainResult`, exposing
  `model` and `state`.
- The entry imports `build_llama_model`; the local test substitutes a tiny
  HF-output double at that construction boundary.
- Tests pass all fields explicitly and use `logging=None`. Concrete logging
  config fields/defaults and logger lifecycle tests remain a later increment.

The entry seeds initialization, builds datasets/loaders, constructs and moves
the model before constructing Muon/AdamW, derives WSD total steps from loader
length, and uses `causal_lm_loss`. Train once through the budget, keeping a
smaller final batch. Enable token counting and LM evaluation perplexity.
Reject training budgets not divisible by sequence length instead of silently
truncating them. Data downloads remain separate.

Checkpoint/resume, distributed training, gradient accumulation, and custom
initialization/output scaling remain separate work.

## Local Red

`./test.sh tests/unit/test_train.py --tb=short` -> exit 1,
**3 failed in 2.79s**. All fail because `RunConfig` is missing from
`upwash.config`. This is intended missing-API Red, not an environment blocker.
Config composition, budget validation, and training assertions have not
executed. No real Llama is constructed by these local tests.

Expected entry example: 21 stored tokens, sequence length 4, budget 20,
batch size 2 -> five samples -> three updates, including the final one-sample
batch, with finite changed parameters. The tiny substitute model uses real
gradients and optimizer updates. No hosted logging occurs.

Regression command:

```bash
./test.sh --ignore=tests/unit/test_train.py \
  --ignore=tests/unit/test_adaptor.py \
  --ignore=tests/unit/components/optimizers/test_muon_adamw.py --tb=short
```

Exit 0, **320 passed in 3.60s**. Existing real-Llama adapter/optimizer files
were not run locally, following the user's sbatch requirement. This is not a
full-suite Green claim. No production, dependency, or test settings changed.

## Slurm Verification — Pending

`scripts/test_train.sbatch` preserves the user's account `ACD115198`, partition
`8gpus`, one node, eight CPUs, one GPU, and four-hour limit. The user selected
the existing project `.venv/bin/python`; the script invokes `./test.sh` via
`srun`, without installing dependencies. Submit from the repository root:

```bash
sbatch scripts/test_train.sbatch
```

The script runs existing real-Llama adapter/optimizer regressions and the
explicit `tests/integration/slurm_train_llama.py` CUDA smoke test. That filename
does not match ordinary pytest discovery, so local default runs do not require
CUDA. No skips, xfails, or pytest configuration changes were added.

The smoke test first runs independent real HF CUDA bf16 forward/backward before
importing the new Upwash API. Reference context: torch 2.15.0+cu132 and
transformers 5.22.2 from the prior recorded environment. It then exercises a
tiny real HF Llama through the entry, Muon/AdamW, WSD, three updates, parameter
changes, and two real dev evaluations (scheduled step 2 and final step 3).
This is a smoke test, not numerical differential training parity. Synthetic
train/dev shards are separate; pretrained downloads and hosted runs are disabled.

`bash -n scripts/test_train.sbatch` passes. `sbatch` is unavailable here; the
job was not submitted. GPU reference execution, smoke assertions, and Slurm
runtime behavior remain unverified. No GPU Red is claimed.

Next: user implements config/result/entry, verify local Green, then submit the
GPU job and inspect its actual output. Later increments cover reproducibility,
logging lifecycle, and numerical WSD wiring.
