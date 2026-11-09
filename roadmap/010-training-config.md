# 010 — Minimal Training Configuration

## Status

Complete for the tested Python-float scope: composed configuration, negative rejection, zero acceptance, and non-finite rejection verified Green on 2026-11-08. Broader accepted numeric types remain deferred. This configuration task is now numbered 010; tasks 001–008 retain their numbers, and 009 is open.

## Goal and Agreed Initial API

Represent the first setting needed for configuration-driven optimizer construction, using composition:

```python
from upwash.config import OptimizerConfig, TrainingConfig

config = TrainingConfig(optimizer=OptimizerConfig(lr=0.1))
assert config.optimizer.lr == 0.1
```

Both arguments are explicit; there is no implicit optimizer or learning-rate choice. Config holds settings, without constructing models, optimizers, or trainers. Proposed validation: accept finite numeric learning rates greater than or equal to zero, reject negative values, NaN, and infinity with `ValueError`. Zero remains valid for controlled no-update experiments. The exact accepted numeric types should be agreed before validation tests.

Only `lr` is currently needed. Data loader settings remain at the existing loader constructor. Add further composed configurations when a concrete construction scenario needs them. YAML, CLI parsing, registries, serialization, device selection, and optimizer-specific options are deferred.

## TDD Sequence

1. Agree on the API and validation scope above.
2. Agent writes one unit test under `tests/unit/` asserting that an explicitly supplied learning rate is accessible through the composed config. With the current empty module, an unavailable public class is the expected initial Red; do not add production stubs.
3. User implements the minimal behavior; agent verifies Green.
4. Add agreed validation cases in separate Red/Green steps, then run the existing regression suite.

No Hugging Face component is being replaced in this task, so no compatibility oracle has been selected.

## Baseline — 2026-11-08

`config.py`, `builder.py`, and `core/logger.py` are empty. `./test.sh` returned `19 passed in 3.70s`. Production code, dependencies, and test configuration were unchanged.

## First Red — 2026-11-08

- Test: `tests/unit/test_training_config.py::test_training_config_exposes_configured_optimizer_learning_rate`.
- Expected behavior: constructing `TrainingConfig(optimizer=OptimizerConfig(lr=0.1))` exposes `config.optimizer.lr == 0.1`.
- Command: `./test.sh tests/unit/test_training_config.py` returned exit 2, one collection error: `ImportError: cannot import name 'OptimizerConfig' from 'upwash.config'`.
- Cause: the production config module is empty and does not yet provide the agreed public classes. This is the intended initial API-availability Red; the value assertion has not executed yet. No production stub was added.
- Existing regression checks: `./test.sh tests/unit/core tests/integration` returned exit 0, `19 passed in 1.91s`. This command covers existing tests and excludes the new Red test; it is not a passing full-suite claim.
- Production code, dependencies, and test configuration remain unchanged.

## Initial Green and Validation Red — 2026-11-08

- User implemented `OptimizerConfig` and `TrainingConfig`; the agent changed no production code.
- Before adding the validation test, `./test.sh` returned exit 0, `20 passed in 1.90s`. The initial composition assertion now executes and passes.
- Next small behavior: constructing `OptimizerConfig(lr=-0.1)` must raise `ValueError`.
- Added `test_optimizer_config_rejects_negative_learning_rate` in `tests/unit/test_training_config.py`.
- `./test.sh tests/unit/test_training_config.py` returned exit 1, `1 failed, 1 passed in 0.01s`. The failure is `DID NOT RAISE ValueError`: the implementation currently stores the negative value unchanged. This is the intended validation Red, not an import or environment failure.
- The full suite has not been rerun after adding this Red test; the `20 passed` result above describes the preceding Green checkpoint.

## Negative Validation Green and Zero Boundary Red — 2026-11-08

- User implemented a guard using `lr <= 0`; the agent changed no production code.
- Before adding the zero-boundary test, `./test.sh` returned exit 0, `21 passed in 1.88s`, confirming negative-rate rejection and existing regressions.
- Added `test_optimizer_config_accepts_zero_learning_rate`, expecting `OptimizerConfig(lr=0.0)` to construct successfully and preserve `lr == 0.0`. This follows the planned nonnegative learning-rate range, including zero for controlled no-update experiments.
- `./test.sh tests/unit/test_training_config.py` returned exit 1, `1 failed, 2 passed in 0.01s`. Construction raises `ValueError` because the current guard also rejects zero: intended boundary Red.
- The full suite has not been rerun after adding this Red test; `21 passed` is the preceding Green checkpoint.

## Zero Boundary Green and Non-finite Validation Red — 2026-11-08

- User changed the guard to `lr < 0` and updated its error text. The agent changed no production code.
- Before adding non-finite cases, `./test.sh` returned exit 0, `22 passed in 1.91s`, confirming zero acceptance and existing regressions.
- Added parametrized `test_optimizer_config_rejects_nonfinite_learning_rate` for Python floats `NaN`, positive infinity, and negative infinity. Each must raise `ValueError`.
- `./test.sh tests/unit/test_training_config.py` returned exit 1, `2 failed, 4 passed in 0.02s`. NaN and positive infinity fail with `DID NOT RAISE ValueError`. Negative infinity already passes through the existing negative-value check. These are intended validation failures, with no import or environment blocker.
- A comparison with zero does not establish finiteness: NaN and positive infinity are currently stored unchanged.
- The full suite has not been rerun after adding this Red test; `22 passed` is the preceding Green checkpoint.

## Final Green — 2026-11-08

- User added a finiteness check using `math.isfinite`, preserving the negative-rate guard. The implementation also explicitly rejects `None`; that additional behavior is not covered by the current tests.
- `./test.sh` returned exit 0, `25 passed in 2.02s`, including all six config cases and existing regression tests.
- Verified scope: composed access to `lr=0.1`, rejection of `-0.1`, acceptance of `0.0`, and rejection of NaN and both infinities. Broader numeric and nonnumeric input types remain deferred; no universal input-type validation guarantee is claimed.
- No agent production edits, dependency changes, or test configuration changes were made.

Next action: agree on the minimal Logger API in [011](011-metric-logger.md) before writing its first Red test.
