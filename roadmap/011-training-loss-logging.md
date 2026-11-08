# 011 — Observe Training Loss Through a Logger

## Status and Dependency

Planned after [010](010-metric-logger.md). Proposed Trainer extension requires agreement before tests or implementation.

## Proposed API and Behavior

Extend the existing constructor with optional `logger=None`:

```python
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, logger=logger)
state = trainer.train(batches)
```

After each successful optimizer update, emit `logger.log({"loss": scalar_loss}, step=completed_steps)`. The value is the loss used for that update, computed before updating parameters, converted to a detached Python number. Keep the gradient path intact through backward. An empty iterable emits no event and keeps the existing `ValueError` behavior. Without a logger, existing training behavior remains available.

Revised proposal on 2026-11-09 for remote backends: keep returned `state.global_step` per-call as already agreed, but use a separate cumulative logging step across calls on the same Trainer. Two one-batch calls would return state steps `1, 1` and emit logging steps `1, 2`. This replaces the earlier unagreed proposal to emit `1, 1`, which cannot be forwarded directly as committed W&B history steps. This distinction requires user agreement before tests. Run resumption, a pre-populated supplied run, and sharing a logger among multiple Trainers need separate design; the initial cumulative-step proposal covers one fresh run and one Trainer. Logger exception handling and recovery after partial training failures also need separate agreement; this task makes no new recovery guarantee.

## TDD Sequence and Numerical Evidence

1. Use a recording test double and one real SGD update: zero-initialized bias-free linear model, input `1`, target `2`, mean MSE, learning rate `0.1`. Expect logged loss `4.0`, step `1`, and final weight approximately `0.4`.
2. Add a second identical batch: losses approximately `4.0, 2.56`, steps `1, 2`, final weight approximately `0.72`. This distinguishes the loss used for an update from loss recomputed afterward.
3. Cover repeated calls and empty input, then run existing no-logger tests and the full suite.

Use float tolerances and real gradients. Do not assert private storage or incidental call order beyond emitting only after a completed update.

No tests run for this feature; no implementation or Red claimed.
