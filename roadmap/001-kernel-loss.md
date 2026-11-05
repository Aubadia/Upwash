# 001 — Compute One Batch's Loss

## Status

Behavior agreed. No test or production implementation has been written. Red and Green have not been established.

## Goal and Agreed Interface

Implement only the kernel behavior needed to calculate one batch's loss:

```text
kernel(model, batch) -> scalar Tensor
batch = (inputs, targets)
loss = MSE(model(inputs), targets), reduction="mean"
```

The concrete callable's public name and import path remain to be agreed before writing its test. A callable is sufficient; this task does not require introducing a class or shared contract.

The loss must ultimately preserve its gradient path as described in [the Trainer goal](../docs/Trainer.md). Task 002 will verify that separately.

## Smallest Acceptance Case

- Execute on CPU with float32 tensors.
- Use `torch.nn.Linear(1, 1, bias=False)` with its weight explicitly initialized to zero.
- Set `inputs = [[1.0]]` and `targets = [[2.0]]` as tensors.
- Call the kernel with the model and `(inputs, targets)`.
- Assert that the result is a zero-dimensional `torch.Tensor` with value approximately `4.0`.

Independent calculation: prediction is `0 * 1 = 0`, so mean squared error is `(0 - 2)^2 = 4`. Use a suitable tolerance, such as `rtol=1e-6` and `atol=1e-6`.

This case establishes the basic scalar loss result. It does not independently distinguish mean from sum reduction on multiple elements; add that case later rather than claiming it is already covered.

## Scope

- Production destination: `src/upwash/components/kernels/`; the user chooses and writes the concrete module and callable.
- Proposed test destination: `tests/unit/components/test_kernel_loss.py`.
- Use the actual kernel and a real PyTorch model. Do not substitute a test-only kernel for the behavior being tested.
- No Trainer, backward call, optimizer update, state, builder, or configuration implementation is needed for this task.
- This is an independently calculated behavioral test, not a Hugging Face differential comparison.

## Next Actions

1. Agree the concrete public name and import path with the user.
2. Recheck the test environment and discuss providing pytest before changing dependencies or test configuration. Do not install packages automatically.
3. Write the smallest acceptance test. Do not create production declarations or stubs to make it importable.
4. Run the test and report its actual failure. Distinguish an import/collection or environment blocker from a numerical behavior failure; do not claim the intended numerical Red until that behavior is reached and fails.
5. Once the intended Red is confirmed, hand implementation to the user.
6. After the user implements the behavior, run the targeted test, record Green if verified, and then move to task 002.

## Test Evidence

- Test command: Not run; the proposed test file does not exist yet.
- Expected result: A scalar tensor approximately equal to `4.0`.
- Observed result: Not available.
- Failure cause: Not established.
- Pending prerequisites: Public name/import path agreement and pytest availability.

