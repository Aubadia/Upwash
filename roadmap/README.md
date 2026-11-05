# Implementation Progress

Last updated: 2026-11-05 (Asia/Taipei).

## Resume Here

Read [the architecture](../docs/Architecture.md), [the Trainer goal](../docs/Trainer.md), and [the current task](001-kernel-loss.md), then inspect the relevant source and tests. Verify this record against the checkout before proceeding.

## Current Position

- **Current task:** [001 — Compute one batch's loss](001-kernel-loss.md).
- **Stage:** Interface agreed; first MSE test written, blocked at collection by the missing production `Trainer` class.
- **Completed:** Agreed `Trainer(model=model, loss_fn=loss_fn)` and `trainer.compute_loss(inputs, targets)`. Trainer holds the model and supplied loss function, computes `loss_fn(model(inputs), targets)`, and initially supports supervised input/target data. The first training loop will unpack `(inputs, targets)` batches. Kernel terminology remains operation-level. Added the user-approved `test.sh` runner and README usage instructions; no dependencies were changed by the agent.
- **Implementation:** The core modules remain empty; the agent has not changed production code. `tests/unit/core/test_trainer_loss.py` now contains the first supplied-MSE test. The supplied-L1 case is pending.
- **Test evidence:** Both `./test.sh` and `../test.sh tests/unit/core/test_trainer_loss.py -k 'compute_loss' -v` (from `tests/`) exited 2 with one collection error: `ImportError: cannot import name 'Trainer' from 'upwash.core.trainer'`. The numerical case did not execute; neither numerical Red nor Green has been established. Runner syntax, help from outside the project, and missing-environment error branches were checked successfully; these checks do not establish task 001 behavior.
- **Environment:** Verified Python 3.14.6, torch 2.15.0, and pytest 9.2.0. Pytest became available during this session after the user elected to prepare it. The agent did not change dependencies or test configuration. Upwash is not installed, so the test command uses `PYTHONPATH=src`. PyTorch also warned about missing NumPy; that warning was not the collection blocker.
- **Next action:** The user provides the production Trainer interface and implementation. Rerun `./test.sh tests/unit/core/test_trainer_loss.py` and report the actual result, then add the L1 case as the next small TDD step. Do not create production stubs or label collection failure as numerical Red.

The user writes production code. The agent supports design, tests, failure analysis, and documentation under [AGENTS.md](../AGENTS.md).

## Incremental Route

Task 001 has an agreed interface and the first test awaiting production implementation. Later rows describe the intended sequence; settle additional interface details when each task is reached.

| Task | Small behavior | Acceptance evidence | Status |
| --- | --- | --- | --- |
| [001](001-kernel-loss.md) | Compute one example's scalar loss using the supplied `loss_fn`. | Scalar loss approximately `4` with MSE and `2` with L1 for the fixed example. | Interface agreed; MSE test collection blocked by missing `Trainer`. L1 test pending. |
| 002 | Preserve the loss's gradient path. | Backward through the returned loss produces model gradient approximately `-4` with supplied MSE loss. | Planned. |
| 003 | Complete one Trainer update. | Real SGD changes the model weight from `0` to approximately `0.4`. | Planned. |
| 004 | Process a second batch with fresh gradients. | A second identical example produces weight approximately `0.72`. | Planned. |
| 005 | Verify training mode and expose progress. | Training executes in training mode; two completed updates return `global_step == 2`. | Planned. |

The numerical update examples in tasks 003 and 004 use an externally supplied MSE loss with mean reduction. Custom operation kernels are not needed for these tasks. Create later task files when their scope is ready, and add boundary and error cases as their expected behavior is agreed.

## Maintaining This Record

After a meaningful change, update the current task and this overview with what actually changed, the exact test command and outcome, any blocker, and the next action. Distinguish implementation present from behavior verified. Do not label a missing dependency, an unrelated failure, or a missing import as evidence that the intended numerical behavior has been exercised.
