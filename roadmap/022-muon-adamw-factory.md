# 022 — Muon and AdamW Factory

## Current Verification — Builder Integration Green

The factory and the user's CombinedOptimizer implementation now pass the
original partition/config test. Baseline command
`./test.sh tests/unit/components/optimizers/test_muon_adamw.py tests/unit/test_builder.py --tb=short`
returned exit 0, `4 passed in 2.87s`.

Added `tests/integration/test_muon_adamw_trainer.py`, injecting
`partial(build_muon_adamw, model_config=model_config)` into the existing
Builder. No Builder changes are needed. A CPU float32 tensor-output tiny LM
with a hidden matrix, bias and tied embedding/head performs two real updates.
Independent real torch 2.15.0+cu132 Muon and AdamW instances run first on a
deep-copied model with explicitly selected groups. Resulting parameters and
last-batch gradients match (`rtol=1e-5`, `atol=1e-7`), all trainable parameters
change, tied identity is preserved and Trainer reports two completed steps.
The reference runs before the Trainer is constructed; inputs and model state
are independent, and initialization preserves caller RNG state.

- `./test.sh tests/integration/test_muon_adamw_trainer.py tests/unit/components/optimizers/test_muon_adamw.py tests/unit/test_builder.py --tb=short`
  → exit 0, **5 passed in 3.08s**.
- `./test.sh --tb=short` → exit 0, **280 passed in 3.42s**.
- The new integration test passed immediately against existing production
  code; no new Red is claimed. This step changed tests and documentation only.

Verified scope: factory construction/grouping and Builder/Trainer two-update
integration, including bias assignment and clearing gradients before the next
batch. The standalone `zero_grad()` result, mutable learning-rate groups,
untied heads, frozen parameters and empty subsets still need dedicated tests.
The integration model emits tensors; direct HF Llama forward-output adaptation
is not covered. Checkpoints, schedules and separate group learning-rate metrics
remain future increments. Next: agree on the next behavior; no current blocker.
The older Red records and next-action notes below are historical.

## Agreed API and Behavior — 2026-11-13

```python
from upwash.components.optimizers.muon_adamw import build_muon_adamw

optimizer = build_muon_adamw(
    model, config=optimizer_config, model_config=model_config,
)
```

Bind `model_config` with `functools.partial` when injecting this factory into
`build_trainer`; the Builder contract from task 021 stays unchanged.

- Muon owns trainable 2D attention/MLP weights.
- AdamW owns embeddings, LM head, normalization parameters and biases.
- Tied embeddings/LM head are assigned once. Every trainable parameter belongs
  to exactly one optimizer.
- Muon uses `lr`, momentum `muon_beta`, and `adjust_lr_fn="match_rms_adamw"`.
- AdamW uses `lr * adam_lr_multiplier` and `(adam_beta1, adam_beta2)`.
- Both use `weight_decay / model_config.d_model`.
- The returned combined optimizer exposes real parameter groups and supports
  `zero_grad()` and `step()` for Trainer. No internal child-optimizer attribute
  names or group ordering are prescribed.

Production location: `components/optimizers/muon_adamw.py` for the factory;
`components/optimizers/combined.py` for the combined optimizer. The agent does
not create production stubs or implementations.

## First Test and Reference

Reference optimizers: `torch.optim.Muon` and `torch.optim.AdamW`, installed
**torch 2.15.0+cu132**. Tiny real Llama construction uses installed
**transformers 5.22.2**, as recorded in task 020. This increment compares
parameter assignments and configured public group settings, not update parity.
Reference instances execute inside the test before the Upwash factory import,
using independent model copies. A standalone CPU reference-construction check
also succeeded before the test was added. No dependency changes.

Test: `tests/unit/components/optimizers/test_muon_adamw.py`.
The real CPU float32 tiny Llama has one layer, width 16, four query heads,
two KV heads and tied embeddings. Seven named attention/MLP matrices belong
to Muon; the embedding and three norm vectors belong to AdamW. The test
compares public group settings to the real references, verifies complete and
unique parameter coverage, and checks construction leaves weights unchanged.
An independent example checks `0.16 / 16 == 0.01`. RNG state is preserved.

## Verified Red

- `./test.sh tests/unit/components/optimizers/test_muon_adamw.py tests/unit/test_builder.py tests/unit/test_adaptor.py --tb=short`
  → exit 1, **1 failed, 4 passed in 3.84s**.
- `./test.sh --tb=short` → exit 1, **1 failed, 278 passed in 3.55s**.
- Sole failure: `ModuleNotFoundError: No module named 'upwash.components'`
  at the agreed factory import. Both reference optimizers successfully
  construct earlier in the test. Upwash group assertions have not executed.
- This is the intended missing-module/API Red, not an environment failure.
  Only tests and roadmap documentation were changed.

Next: user implements the factory and combined optimizer, then rerun for
Green. Separate later tests will exercise actual Muon/AdamW updates through
the combined object, gradient clearing, biases, frozen parameters, untied
heads and empty subsets. Checkpoint state, scheduler integration and per-group
learning-rate logging remain later work. Current Trainer logs only the first
group's learning rate.
