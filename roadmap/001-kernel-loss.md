# 001 — Compute One Batch's Loss

## Status

Interface agreed on 2026-11-05. The first supplied-MSE test has been written and run, but collection fails because the empty production module does not define `Trainer`. Numerical Red and Green have not been established. The supplied-L1 case remains the next small test after the MSE case is exercised.

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

1. The user provides the production `Trainer` interface. The agent must not create declarations or stubs to make the test importable.
2. Rerun the supplied-MSE test when the interface exists. Report the actual result; collection failure is not numerical Red. The user may implement the full behavior directly, in which case report any verified Green without claiming a prior numerical Red.
3. After the MSE case is exercised, add the supplied-L1 case as the next small TDD step to verify that the externally supplied loss function controls the result.
4. Once both cases pass, record task 001 as complete and move to task 002 for gradient verification.

## Test Evidence

- Convenient runner: `./test.sh tests/unit/core/test_trainer_loss.py`. With no arguments, `./test.sh` runs all tests. The user approved this script; it uses the existing virtual environment and installs no dependencies. See the root README for usage.
- Test command: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/core/test_trainer_loss.py -q`.
- Failing case: `test_compute_loss_uses_supplied_mse_loss`; the case did not execute because collection failed.
- Expected result: A zero-dimensional `torch.Tensor` approximately equal to `4.0` with supplied MSE loss, using `rtol=1e-6` and `atol=1e-6`.
- Observed result: Exit code 2, one collection error: `ImportError: cannot import name 'Trainer' from 'upwash.core.trainer'`.
- Failure cause: `src/upwash/core/trainer.py` is empty. The model/loss numerical behavior has not been exercised, so this is not numerical Red.
- Environment: Python 3.14.6, torch 2.15.0, pytest 9.2.0. Pytest was initially absent and became available during this session after the user elected to prepare it. The agent did not install packages or modify dependency/test configuration. The command uses `PYTHONPATH=src` because Upwash is not installed in the environment.
- Additional output: PyTorch warned that NumPy is unavailable; the blocking error is the missing `Trainer`, not that warning. No NumPy dependency has been added by the agent.
- Pending prerequisite: User-written production interface and implementation. The L1 test has not yet been written or run; no tests have passed.

### Runner Verification (2026-11-05)

- `sh -n test.sh`: passed shell syntax validation.
- `./test.sh`: exit 2, the same missing-`Trainer` collection error described above.
- `../test.sh tests/unit/core/test_trainer_loss.py -k 'compute_loss' -v` from `tests/`: exit 2, the same collection error. Project-root resolution and argument forwarding worked; the selected test body could not execute.
- Absolute-path invocation with `--help` from `/private/tmp`: exit 0, pytest help displayed.
- Temporary isolated copies verified exit 1 and a clear message for a missing `.venv/bin/python` and for simulated pytest import failure. The actual project environment was not modified.
- These are runner checks, not numerical Red or Green for loss computation.
