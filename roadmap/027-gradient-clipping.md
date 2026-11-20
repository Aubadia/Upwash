# 027 — Gradient clipping

Branch `feat/grad_clip`, based on `main` at `73b1875`. The behavior comes
from `experiment` commit `7598e33`.

## Agreed behavior

- `TrainingConfig(..., max_grad_norm=None)` passes the threshold through
  `build_trainer` to `Trainer`. The default disables clipping. A supplied
  threshold must be finite and positive.
- After backward and before the optimizer update, clip the global L2 norm
  across model parameters that have gradients. With DDP, use the synchronized
  gradients. Clipping applies with or without a logger.
- `train/grad_norm` reports the pre-clip norm.

## First Red — 2026-11-20

The CPU integration test builds the Trainer through its usual config path,
runs one real SGD update, and checks the weight, bias, and logged norm.
The raw gradients are -3 and -4, with a joint norm of 5. A threshold of 1
and learning rate of 0.1 give parameter values of approximately 0.06 and
0.08. The logged norm is 5.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/integration/test_gradient_clipping.py tests/integration/test_training_logging.py tests/unit/test_training_config.py tests/unit/test_builder.py tests/unit/core/test_trainer_world_size.py
```

Exit 1, **1 failed, 77 passed in 11.92s**. `TrainingConfig.__init__` rejects
the new `max_grad_norm` keyword with `TypeError`. The test has yet to reach the
update and logging assertions.

Next, implement config validation, builder forwarding, and clipping in the
Trainer, then rerun the same tests. The first test covers the logging-enabled
path. The disabled, no-logger, and invalid-threshold cases are still unverified.

## Verification after the first Trainer implementation — 2026-11-20

The same regression command exited 1 with **9 failed, 69 passed in 12.77s**.
The new integration still stops at the unsupported
`TrainingConfig(max_grad_norm=...)` keyword. Eight existing training-logging
cases now fail because Trainer enables clipping by default with a limit of
1.0. For example, the first logged mean loss is 3.805 instead of 3.28.

A direct single-step CPU Trainer probe with the same model, data, SGD learning
rate, and explicit limit of 1 confirmed the weight and bias updates
(approximately 0.06 and 0.08). It logged approximately 1.0 where the expected
pre-clip norm is 5.0. Trainer currently discards the norm returned by
`clip_grad_norm_` and computes the logged norm from the clipped gradients.

Next, make clipping opt-in, pass the validated threshold from config through
the builder, and use the norm returned by the clipping call for logging.
Then rerun the regression command above.

## Verification after config and builder wiring — 2026-11-20

The same regression command exited 1 with **1 failed, 77 passed in 11.89s**.
Clipping is now opt-in, config reaches Trainer through the builder, and both
parameter update assertions pass. The sole failure is the metric assertion:
`train/grad_norm` is approximately 1.0, with 5.0 expected. The clipping call's
return value is still discarded. The config currently stores the threshold
without checking that it is finite and positive.

Next, use the norm returned by the clipping call for the metric, validate the
threshold in config, and rerun the targeted regressions.

## Clipped update Green and boundary Red — 2026-11-20

The same regression command exited 1 with **6 failed, 78 passed in 5.31s**.
The original integration now passes, including the pre-clip norm. The six
failures are new boundary cases:

- With clipping disabled, two finite FP32 gradients of 1e30 should yield SGD
  weights of approximately -1e29. Instead both weights become NaN. The
  infinite norm overflows, and using an infinite clipping threshold yields
  an indeterminate scale.
- The five invalid thresholds (zero, a negative number, NaN, and both
  infinities) are accepted without the expected ValueError.

Next, bypass clipping when the threshold is None and compute the norm only
when logging needs it. Validate supplied thresholds in TrainingConfig, then
rerun the same regression command.

## Green — 2026-11-20

The regression command above now exits 0 with **84 passed in 5.49s**.
Trainer bypasses clipping when the threshold is None, and TrainingConfig
rejects the five invalid thresholds.

The expanded CPU regression exits 0 with **87 passed in 32.16s**. It adds
an SGD update without a logger and runs the existing two-rank Gloo versus
single-process comparison with and without clipping. The clipped case
checks that the single-process final gradient norm reaches the configured
limit of 0.1 and that both ranks match the single-process update within the
existing tolerance.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/integration/test_gradient_clipping.py tests/integration/test_training_logging.py tests/unit/test_training_config.py tests/unit/test_builder.py tests/unit/core/test_trainer_world_size.py tests/integration/test_ddp_training.py
```

The full suite and GPU runs were not repeated for this task. The targeted
CPU verification is complete.
