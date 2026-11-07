# 003 — Model Forward Hooks

## Behavior

Exercise the existing `compute_loss(inputs, targets)` contract through the documented `model(inputs)` call. A model's registered forward hook may replace its output; the supplied loss function must receive that output. No constructor or training API changes are needed.

The CPU float32 test uses a zero-weight, bias-free linear model, input `1`, target `2`, and mean MSE. A forward hook adds `1` to the prediction. Expected loss is `(1 - 2)^2 = 1`, rather than the unmodified loss `4`. The test removes the hook in a `finally` block and restores RNG state after model initialization. No dependencies or test configuration changed.

## Verified Red — 2026-11-07

- `./test.sh tests/unit/core/test_trainer_loss.py::test_compute_loss_uses_output_modified_by_model_forward_hook`: exit 1, `1 failed in 2.02s`.
- `./test.sh`: exit 1, `1 failed, 5 passed in 2.04s`.
- Expected loss `1`; observed loss `4`. `compute_loss` directly calls `self.model.forward(inputs)`, bypassing the model's registered forward hook. This is the intended numerical failure; all previous tests remain Green.

## Next Action

The user changes model invocation to `self.model(inputs)`, then the agent reruns the targeted test and full suite. The agent has not modified production code. This test covers output replacement by a forward hook, without claiming comprehensive hook or Hugging Face compatibility.
