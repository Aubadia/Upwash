# 001 — Compute One Batch's Loss

## Status

Behavior agreed. No test or production implementation has been written. Red and Green have not been established.

## Goal and Agreed Interface

Implement only the kernel behavior needed to calculate one batch's loss using a required, externally supplied loss function:

```text
kernel(model, batch) -> scalar Tensor
batch = (inputs, targets)
predictions = model(inputs)
loss = loss_fn(predictions, targets)
```

Supply `loss_fn` when constructing or configuring the kernel; do not assume MSE or provide a default loss function. For this first supervised kernel, `loss_fn` accepts predictions and targets and returns a scalar tensor. The supplied loss function controls reduction; the kernel does not impose mean reduction. The Trainer's target API remains unchanged and does not receive `loss_fn` directly.

The concrete callable's public name, import path, and construction/configuration syntax for supplying `loss_fn` remain to be agreed before writing its test. A callable is sufficient; this task does not require introducing a class or shared contract.

The loss must ultimately preserve its gradient path as described in [the Trainer goal](../docs/Trainer.md). Task 002 will verify that separately.

## Small Acceptance Cases

- Execute on CPU with float32 tensors.
- Use `torch.nn.Linear(1, 1, bias=False)` with its weight explicitly initialized to zero.
- Set `inputs = [[1.0]]` and `targets = [[2.0]]` as tensors.
- Configure the actual kernel with the supplied loss function and call it with the model and `(inputs, targets)`.
- First supply `torch.nn.MSELoss(reduction="mean")` and assert that the result is a zero-dimensional `torch.Tensor` with value approximately `4.0`.
- Then add a separate case supplying `torch.nn.L1Loss(reduction="mean")` with the same model state and inputs, and assert that the result is a zero-dimensional `torch.Tensor` with value approximately `2.0`. This checks that the kernel uses the supplied loss function rather than always computing MSE.

Independent calculation: prediction is `0 * 1 = 0`, so squared error is `(0 - 2)^2 = 4` and absolute error is `abs(0 - 2) = 2`. Use a suitable tolerance, such as `rtol=1e-6` and `atol=1e-6`.

MSE and L1 are test inputs, not built-in kernel choices. These single-element cases do not independently distinguish mean from sum reduction on multiple elements; add that case later rather than claiming it is already covered.

## Scope

- Production destination: `src/upwash/components/kernels/`; the user chooses and writes the concrete module and callable.
- Proposed test destination: `tests/unit/components/test_kernel_loss.py`.
- Use the actual kernel and a real PyTorch model. Do not substitute a test-only kernel for the behavior being tested.
- No Trainer, backward call, optimizer update, state, builder, or configuration implementation is needed for this task.
- This is an independently calculated behavioral test, not a Hugging Face differential comparison.

## Next Actions

1. Agree the concrete public name, import path, and how `loss_fn` is supplied with the user.
2. Recheck the test environment and discuss providing pytest before changing dependencies or test configuration. Do not install packages automatically.
3. Start with the supplied-MSE acceptance test, then add the supplied-L1 case as a separate small TDD step. Do not create production declarations or stubs to make tests importable.
4. Run the test and report its actual failure. Distinguish an import/collection or environment blocker from a numerical behavior failure; do not claim the intended numerical Red until that behavior is reached and fails.
5. Once the intended Red is confirmed, hand implementation to the user.
6. After the user implements the behavior, run the targeted test, record Green if verified, and then move to task 002.

## Test Evidence

- Test command: Not run; the proposed test file does not exist yet.
- Expected result: A scalar tensor approximately equal to `4.0` with supplied MSE loss, or `2.0` with supplied L1 loss.
- Observed result: Not available.
- Failure cause: Not established.
- Pending prerequisites: Public name/import path and loss-function injection syntax agreement, and pytest availability.

