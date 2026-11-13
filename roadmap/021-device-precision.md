# 021 — Device Placement and bf16 Mixed Precision

## Behavior

Let Trainer run on a GPU with bf16 mixed precision. Today nothing places tensors on a device: data, model, and computation stay on the CPU, and a model moved to CUDA fails on CPU batches.

Branch `feat/gpu-bf16`, from `main` at `c37845f` (PR #8 merged). Task number 021 because 020 is reserved for the Muon + AdamW builder. The duplicated 017–019 numbers from the logging work are still to be renumbered.

## Agreed Interface — 2026-11-13

```python
model = model.to("cuda")                     # the entry script decides the device
config = TrainingConfig(optimizer=..., precision="bf16")
trainer = build_trainer(config, model, loss_fn, optimizer_factory)   # signature unchanged
```

- Device is inferred from the model. Trainer moves each `(inputs, targets)` batch to the device of the model's parameters, in both `train` and `evaluate`. There is no `device` argument, so the model placement and the batch placement can't disagree. DDP- or FSDP-wrapped models already have their parameters on the rank's device.
- `precision` is `"fp32"` (default) or `"bf16"`.
  - `"bf16"` is mixed precision. `compute_loss` (forward and loss) runs under `torch.autocast(device_type=<model device type>, dtype=torch.bfloat16)`. Backward and the optimizer step run outside autocast. Parameters, gradients, and optimizer state stay `float32`. No `GradScaler` is needed for bf16.
  - `"fp32"` uses no autocast context at all. `torch.autocast` rejects device types it doesn't support, such as `meta`, even with `enabled=False`.
  - Trainer validates `precision` and raises `ValueError` for anything else. Tests construct Trainer directly, so validation lives where the value is used. Config-level validation comes when `TrainingConfig` becomes a dataclass.
- `TrainingConfig(optimizer, logging_steps=None, eval_steps=None, precision="fp32")`. Precision affects results, so it belongs in the recorded run configuration. `build_trainer` passes `config.precision` to Trainer.

Deferred:

- `torch.cuda.synchronize()` (or CUDA events) for accurate `train/tokens_per_second` on GPU. That changes the interval timing in `_LoggingInterval` and needs its own cost/benefit design.
- A CUDA-only pytest marker. GPU behavior is checked by a smoke run on nano4 instead.
- Pinned memory and `non_blocking` transfers. Our `DataLoader` doesn't pin memory yet.

## Tests

`tests/unit/core/test_trainer_device.py`. A CPU-only machine has no second real device, so the `meta` device stands in for an accelerator. A meta model can't consume CPU tensors, and a forward hook and the loss function record the devices they receive.

- `train` delivers inputs and targets on `meta` for each of two CPU batches.
- `evaluate` does the same. The loss function returns a real CPU scalar, because meta tensors have no values for `.item()`.

`tests/unit/core/test_trainer_precision.py`. CPU autocast supports bfloat16. A forward hook records the `nn.Linear` output dtype.

- `train` and `evaluate` produce `bfloat16` outputs for `"bf16"` and `float32` outputs for `"fp32"`.
- The default is fp32.
- After bf16 training, the weight and its gradient are `float32` and the weight has changed.
- Autocast doesn't leak: after a bf16 `train`, a new `nn.Linear` produces `float32`.
- `"fp16"`, `"BF16"`, `"bfloat16"`, `"float32"`, and `""` raise `ValueError`.

`tests/integration/test_bf16_training.py`. Real SGD on a 4-input linear regression, 20 updates.

- fp32 and bf16 both cut the evaluated loss by more than 10× (about 15.2 → 0.29).
- bf16 weights and bias match fp32 within `atol=0.02`, and the final losses agree within 0.02. bfloat16 keeps 8 significand bits, and the observed weight drift is about 0.003. fp32 itself moves weights by more than 1, so a broken update still fails.

`tests/unit/test_builder.py`: `TrainingConfig(precision="bf16")` gives a trainer whose forward runs in `bfloat16`.

`tests/unit/test_training_config.py`: `precision` defaults to `"fp32"`, and `"bf16"` is stored.

## Verified Red — 2026-11-13

Environment checks (Docker, `upwash-venv-pr5`, torch `2.15.0+cu132`, CPU):

- A CPU input with a meta `nn.Linear` (with bias) raises `Tensor on device meta is not on the expected device cpu!`, and a full SGD step on meta tensors works.
- CPU bf16 autocast gives `bfloat16` linear outputs. `mse_loss` and `cross_entropy` return `float32`, and parameters stay `float32`.
- `torch.autocast(device_type="meta", ...)` raises `RuntimeError`, with or without `enabled=False`.

Command: `PYTHONPATH=src uv run pytest -q tests/unit/core/test_trainer_device.py tests/unit/core/test_trainer_precision.py tests/integration/test_bf16_training.py tests/unit/test_builder.py tests/unit/test_training_config.py`.

Result: `17 failed, 46 passed`. Of the 18 new tests, 17 fail and 1 (`test_precision_defaults_to_fp32`) already passes, because today's Trainer runs fp32 when no precision is given.

- `test_train_moves_each_batch_to_the_model_device`: `RuntimeError: Function MmBackward0 returned an invalid gradient at index 1 - expected device meta but got cpu`. The CPU batch reached the meta model unmoved: a bias-free meta linear accepts the CPU input in forward, and backward rejects the mixed devices.
- `test_evaluate_moves_each_batch_to_the_model_device`: `assert ['cpu', 'cpu'] == ['meta', 'meta']`.
- Thirteen precision cases: `TypeError: Trainer.__init__() got an unexpected keyword argument 'precision'`. These are the five train/evaluate dtype cases, the float32-parameters case, the no-leak case, five invalid values, and the integration case.
- Builder and config cases: `TrainingConfig.__init__() got an unexpected keyword argument 'precision'` and `'TrainingConfig' object has no attribute 'precision'`.

All failures come from the missing behavior or API. No environment failure occurred.

Rest of the suite, ignoring the three new files and deselecting the new precision cases: `276 passed`.

Test sanity check: a scratch reference implementation outside the repository passed the full suite (`294 passed`). It inferred the device from `next(model.parameters())`, moved both tensors in train and evaluate, used autocast for bf16 or `nullcontext` for fp32, and passed `precision` through the config and builder.

## Implementation Review — 2026-11-13

The user's first implementation added `precision` to Trainer, `TrainingConfig`, and `build_trainer`. It added `_to_model_device` and `_autocast` helpers, and moved batches and wrapped `compute_loss` in both `_training_step` and `evaluate`.

Full suite (Docker, `upwash-venv-pr5`): `5 failed, 289 passed`. All five failures are the invalid-precision cases, `Failed: DID NOT RAISE ValueError`, because the precision isn't validated yet.

Review also found a defect the CPU tests could not see. `_autocast` called `torch.autocast(device_type=device.type)` without `dtype`. Autocast then uses the device default, which is `torch.bfloat16` on CPU but `torch.float16` on CUDA (checked with `torch.get_autocast_dtype`). On nano4, "bf16" would silently run float16 without a `GradScaler`.

Added a regression test, `test_bf16_means_bfloat16_even_when_the_device_default_is_float16` (train and evaluate). A fixture sets the CPU autocast default to float16 to reproduce the CUDA default, and restores the previous value afterwards.

- Red against the user's code: `assert [torch.float16] == [torch.bfloat16]` in both cases.
- A scratch copy with `dtype=torch.bfloat16` added: `2 passed`.

## Verified Green — 2026-11-13

The user added both fixes:

- Trainer raises `ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")` for any other value.
- `_autocast` returns `torch.autocast(device_type=device.type, dtype=torch.bfloat16)` for bf16, and `nullcontext()` otherwise.

Command (Docker, `upwash-venv-pr5`): `uv run pytest -q` returned `296 passed`. That is the previous 276 plus the 20 tests of this task.

Optional refactors suggested to the user, none of which change behavior:

- Validate before setting any attribute.
- Extract a `_device()` helper used by both helpers.
- Drop the `else`.
- Remove whitespace-only lines in `trainer.py` and the blank line in the `build_trainer` signature.
- Add a trailing comma after `precision=config.precision`.

Not verified: real CUDA execution. CPU tests can't show the speed or memory effect of bf16 on a GPU, or device transfer to `cuda`. Check these in the first nano4 smoke run: a bf16 run should report higher `train/tokens_per_second` and lower memory than fp32.

Status: complete in code and tests. Not yet committed. Next: commit, then confirm on GPU during the smoke run. The deferred items above, GPU timing synchronization and pinned memory, remain open.
