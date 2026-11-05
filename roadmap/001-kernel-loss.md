# 001 — Compute One Batch's Loss

## Status

Completed and verified on 2026-11-05. The user's `Trainer.compute_loss` now runs the model before applying the externally supplied loss function. Both agreed scalar-loss cases pass: MSE returns approximately `4.0`, and L1 returns approximately `2.0`. MSE previously had a confirmed numerical Red; L1 passed on its first run against the existing implementation. Gradient behavior remains for task 002.

The filename is retained to preserve existing links. "Kernel" in this project means an operation implementation such as matmul or attention. The earlier training-computation kernel interface is withdrawn; this task does not implement an operation kernel.

## Goal

Establish the smallest Upwash behavior that computes a scalar loss using a real model and an externally supplied loss function.

The agreed public interface is:

```python
from upwash.core.trainer import Trainer

trainer = Trainer(model=model, loss_fn=loss_fn)
loss = trainer.compute_loss(inputs, targets)
```

Trainer holds the externally supplied model and loss function. The first supervised flow uses explicit input and target tensors. Later, Trainer will unpack `(inputs, targets)` batches before calling this method. This is the initial supported format, not a universal batch format. Other calling conventions and optimizer injection remain deferred.

The numerical calculation is:

```python
predictions = model(inputs)
loss = loss_fn(predictions, targets)
```

`compute_loss` performs this computation and returns the loss tensor. The loss function is externally supplied, with no default MSE. Its configuration controls reduction, and it must return a scalar tensor for this initial training flow.

The loss must ultimately preserve its gradient path as described in [the Trainer goal](../docs/Trainer.md). Task 002 will verify that separately.

## Agreed Small Acceptance Cases

Through the agreed Upwash interface:

- Execute on CPU with float32 tensors.
- Use `torch.nn.Linear(1, 1, bias=False)` with its weight explicitly initialized to zero.
- Set `inputs = [[1.0]]` and `targets = [[2.0]]` as tensors.
- First supply `torch.nn.MSELoss(reduction="mean")` through the agreed interface and assert a zero-dimensional `torch.Tensor` with value approximately `4.0`.
- Then add a separate case supplying `torch.nn.L1Loss(reduction="mean")` with the same model state and inputs, and assert a zero-dimensional `torch.Tensor` with value approximately `2.0`.

Independent calculation: prediction is `0 * 1 = 0`, so squared error is `(0 - 2)^2 = 4` and absolute error is `abs(0 - 2) = 2`. Use a suitable tolerance, such as `rtol=1e-6` and `atol=1e-6`.

MSE and L1 are test inputs, not built-in training choices. These single-element cases do not independently distinguish mean from sum reduction on multiple elements. A test that only invokes PyTorch's model and loss function would establish a baseline, not verify an Upwash implementation.

## Scope

- Production destination: `src/upwash/core/trainer.py`, class `Trainer`, method `compute_loss(inputs, targets)`. Test destination: `tests/unit/core/test_trainer_loss.py`. Do not place model-and-batch orchestration in `components/kernels/`.
- Use the actual agreed Upwash behavior and a real PyTorch model. Do not substitute test support code for the behavior being tested.
- No custom matmul, attention, or other operation-kernel implementation is needed.
- Do not introduce a replacement training-step abstraction solely to preserve the previous design.
- Backward, optimizer updates, state, and builder implementation belong to later steps.
- This is an independently calculated behavioral test, not a Hugging Face differential comparison.

## Next Actions

1. Begin task 002: verify that backward through the supplied-MSE loss produces model weight gradient approximately `-4.0` for the same fixed example. No new production interface is required.
2. Suggested user-owned cleanup: call `self.model(inputs)` instead of `self.model.forward(inputs)` so that execution uses the standard `nn.Module` call path. The task 001 tests verify loss values, not model hooks.

## Test Evidence

- Convenient runner: `./test.sh tests/unit/core/test_trainer_loss.py`. With no arguments, `./test.sh` runs all tests. The user approved this script; it uses the existing virtual environment and installs no dependencies. See the root README for usage.
- MSE verification before adding L1: `./test.sh tests/unit/core/test_trainer_loss.py` exited 0 with `1 passed`.
- Full suite after adding L1: `./test.sh` exited 0 with `2 passed`, no warnings.
- Passing cases: `test_compute_loss_uses_supplied_mse_loss` and `test_compute_loss_uses_supplied_l1_loss`.
- Expected and verified results: Zero-dimensional `torch.Tensor` values approximately `4.0` with MSE and `2.0` with L1, using `rtol=1e-6` and `atol=1e-6`.
- Earlier numerical Red: The MSE test exited 1 with `Expected 4.0 but got 1.0` when the implementation passed raw inputs directly to the loss function. The user's subsequent implementation computes predictions first and passes this test. L1 passed immediately; no L1 Red is claimed.
- Environment: Python 3.14.6, torch 2.15.0, pytest 9.2.0, NumPy 2.5.4. The user supplied pytest and added NumPy; the agent did not install packages or change dependency configuration. `test.sh` supplies `PYTHONPATH=src` and uses the existing virtual environment.
- Remaining verification: Backward and model gradients have not been tested. Task 001 is complete within its scalar-loss scope; task 002 is next.

### Earlier Runner Verification (2026-11-05, Before the User's Implementation)

- `sh -n test.sh`: passed shell syntax validation.
- `./test.sh`: exit 2, `ImportError: cannot import name 'Trainer' from 'upwash.core.trainer'` before the production class existed.
- `../test.sh tests/unit/core/test_trainer_loss.py -k 'compute_loss' -v` from `tests/`: exit 2, the same collection error. Project-root resolution and argument forwarding worked; the selected test body could not execute at that stage.
- Absolute-path invocation with `--help` from `/private/tmp`: exit 0, pytest help displayed.
- Temporary isolated copies verified exit 1 and a clear message for a missing `.venv/bin/python` and for simulated pytest import failure. The actual project environment was not modified.
- These are runner checks, not numerical Red or Green for loss computation.
