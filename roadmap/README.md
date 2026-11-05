# Implementation Progress

Last updated: 2026-11-05 (Asia/Taipei).

## Resume Here

Read [the architecture](../docs/Architecture.md), [the Trainer goal](../docs/Trainer.md), and [the current task](001-kernel-loss.md), then inspect the relevant source and tests. Verify this record against the checkout before proceeding.

## Current Position

- **Latest completed task:** [001 — Compute one batch's loss](001-kernel-loss.md).
- **Stage:** Task 001 Green verified. Next: task 002, preserve the loss's gradient path; its test has not yet been written.
- **Completed:** Agreed `Trainer(model=model, loss_fn=loss_fn)` and `trainer.compute_loss(inputs, targets)`. Trainer holds the model and supplied loss function, computes `loss_fn(model(inputs), targets)`, and initially supports supervised input/target data. The first training loop will unpack `(inputs, targets)` batches. Kernel terminology remains operation-level. Added the user-approved `test.sh` runner and README usage instructions; no dependencies were changed by the agent.
- **Implementation:** The user has implemented `Trainer(model, loss_fn)` and `compute_loss(inputs, targets)` in `core/trainer.py`. The method now invokes `self.model.forward(inputs)` and applies the supplied loss function to predictions and targets. The agent has not changed production code. `tests/unit/core/test_trainer_loss.py` contains separate MSE and L1 cases.
- **Test evidence:** Before adding L1, `./test.sh tests/unit/core/test_trainer_loss.py` exited 0 with `1 passed`. After adding L1, `./test.sh` exited 0 with `2 passed`, no warnings. MSE verifies scalar loss approximately `4`; L1 verifies scalar loss approximately `2`. The prior MSE numerical Red is recorded in task 001. L1 passed on its first run; no L1 Red is claimed. Backward has not been verified.
- **Environment:** Verified Python 3.14.6, torch 2.15.0, pytest 9.2.0, and NumPy 2.5.4. The user supplied pytest and added NumPy; the agent did not change dependency configuration. `test.sh` uses the existing `.venv` and supplies `PYTHONPATH=src`.
- **Next action:** Start task 002 with the existing public interface: backward through the supplied-MSE loss must produce model weight gradient approximately `-4.0` for the fixed example. Suggested user-owned cleanup: use `self.model(inputs)` rather than directly calling `.forward(inputs)`; model hook behavior is not covered by task 001.

The user writes production code. The agent supports design, tests, failure analysis, and documentation under [AGENTS.md](../AGENTS.md).

## Incremental Route

Task 001 is complete within its scalar-loss scope. Task 002 is next. Later rows describe the intended sequence; settle additional interface details when each task is reached.

| Task | Small behavior | Acceptance evidence | Status |
| --- | --- | --- | --- |
| [001](001-kernel-loss.md) | Compute one example's scalar loss using the supplied `loss_fn`. | Scalar loss approximately `4` with MSE and `2` with L1 for the fixed example. | Complete: both cases passed in `./test.sh`. |
| 002 | Preserve the loss's gradient path. | Backward through the returned loss produces model gradient approximately `-4` with supplied MSE loss. | Next; test not yet written. |
| 003 | Complete one Trainer update. | Real SGD changes the model weight from `0` to approximately `0.4`. | Planned. |
| 004 | Process a second batch with fresh gradients. | A second identical example produces weight approximately `0.72`. | Planned. |
| 005 | Verify training mode and expose progress. | Training executes in training mode; two completed updates return `global_step == 2`. | Planned. |

The numerical update examples in tasks 003 and 004 use an externally supplied MSE loss with mean reduction. Custom operation kernels are not needed for these tasks. Create later task files when their scope is ready, and add boundary and error cases as their expected behavior is agreed.

## Maintaining This Record

After a meaningful change, update the current task and this overview with what actually changed, the exact test command and outcome, any blocker, and the next action. Distinguish implementation present from behavior verified. Do not label a missing dependency, an unrelated failure, or a missing import as evidence that the intended numerical behavior has been exercised.
