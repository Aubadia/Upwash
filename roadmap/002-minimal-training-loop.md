# 002 — Minimal Training Loop

## Status

2026-11-07: Complete for the agreed minimal-loop and per-call progress scope.

The user agreed on per-call progress and fixed the counter initialization: `train()` resets progress and initializes the local `steps` to integer zero. The agent did not modify production code.

Verified on 2026-11-07:

- `./test.sh tests/integration/test_minimal_training_loop.py::test_train_counts_steps_per_call_and_continues_updating_parameters`: exit 0, `1 passed in 2.02s`.
- `./test.sh`: exit 0, `5 passed in 2.01s`.
- Two successive one-batch calls return immediate `global_step` values `1 → 1`, with real SGD weights approximately `0.4 → 0.72`. Existing scalar-loss, training-mode, and fresh-gradient scenarios remain Green.
- The earlier `NoneType + int` counter regression is resolved. Its prior full-suite result was `3 failed, 2 passed in 1.85s`; the repeated-call test then failed before reaching its assertions.
- Next action: select the next concrete training requirement with the user. Empty-input, error-state, and returned-state snapshot semantics remain deferred. Suggested user-owned cleanup remains calling `self.model(inputs)` instead of `.forward(inputs)`; hook behavior has not been tested.

## Agreed Interface

```python
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
state = trainer.train(batches)
```

The caller constructs a real `torch.optim.SGD` over the model parameters. `batches` is an iterable of `(inputs, targets)` pairs. The loop traverses it once, sets model training mode, clears gradients for each batch, computes scalar loss, runs backward, and steps the supplied optimizer. The returned object exposes `global_step`, counting completed updates in that call; its concrete type is not part of this proposal.

The user directed the agent to correct the loss tests to supply the optimizer required by the current constructor. This supersedes the earlier optional-optimizer proposal. Loss tests construct a real SGD optimizer but do not invoke training or change their numerical expectations.

## Acceptance Test Plan

Use small real CPU float32 models and SGD with learning rate `0.1`, no momentum or weight decay. With a bias-free linear model initialized to weight zero, input `[[1.0]]`, target `[[2.0]]`, and supplied mean MSE:

- One batch produces weight approximately `0.4` and `global_step == 1`.
- Two identical batches produce weight approximately `0.72` and `global_step == 2`, exercising fresh gradients and real backward computation.
- Two successive one-batch calls each return `global_step == 1`, while weight continues from approximately `0.4` to `0.72`.
- Start the model in evaluation mode and observe training mode during its forward execution using test support code.

Use `rtol=1e-6`, `atol=1e-6` for numerical comparisons. Place loop tests under `tests/integration/`. These are independently calculated acceptance tests, with no Hugging Face compatibility claim or new dependencies. The test-only linear model records mode during real forward computation. The tests supply batches through an iterator and restore CPU RNG state after model initialization.

## Scope

No builder, configuration, new shared contracts, custom kernels, epochs, device management, scheduler, logging, or checkpointing is needed. Per-call repeated-call progress is agreed and verified Green. Empty-input, error-state, and returned-state snapshot behavior remain deferred. Existing scalar-loss tests remain regression coverage. See [Trainer goal](../docs/Trainer.md) for the numerical derivation and agreed destination.

## Earlier Test Evidence (Before User Implementation)

- Added explanatory comments for RNG isolation, initial evaluation mode, iterable input, the SGD calculation, gradient accumulation, and forward-mode observations. This comments-only edit was not rerun; the execution evidence below predates it.
- `./test.sh tests/integration/test_minimal_training_loop.py`: exit 1, `2 failed in 3.85s`.
- `./test.sh`: exit 1, `2 failed, 2 passed in 1.91s`; existing scalar-loss cases remain Green.
- Both new cases fail with `TypeError: Trainer.__init__() got an unexpected keyword argument 'optimizer'`.
- Expected: weights approximately `0.4` and `0.72`, completed step counts `1` and `2`, and training mode active during forward. Observed: execution stops before `train` is invoked. This confirms a missing agreed API; no numerical Red or verified backward behavior is claimed.
- At that stage, implementation was handed to the user. This historical failure is superseded by the passing suite recorded above; no production stubs were added.
