# 021 — Model-Aware Optimizer Factory

## Green — 2026-11-13

The user implemented `optimizer_factory(model, config=config.optimizer)`.
The real Muon construction test now passes, including model/config identity,
optimizer injection and configured settings. Migrated legacy SGD test callers
to the approved contract through a shared test-only `sgd_optimizer_factory`
fixture; their numerical, logging and evaluation expectations are unchanged.
No agent production edits or dependency/test configuration changes.

- Single Muon test with `--tb=short`: exit 0, `1 passed in 2.22s`.
- `./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py tests/unit/test_training_config.py tests/integration/test_training_logging.py tests/integration/test_scheduled_evaluation.py`:
  exit 0, `72 passed in 2.37s`.
- `./test.sh`: exit 0, `278 passed in 3.30s`.

This completes the agreed factory-injection increment. No current blocker.
Next: agree on the concrete Muon/AdamW factory API and parameter grouping,
including tied embeddings, norms and width-scaled weight decay, before its
first Red test. The Muon factory currently exists only inside the test;
production optimizer construction and Muon updates through Trainer are not
implemented or verified by this increment. Earlier Red records below are
historical.

## Agreed First Increment — 2026-11-13

The user approved changing the injected factory contract to
`optimizer_factory(model, *, config: OptimizerConfig)` and requested Muon in
the test. Builder passes the supplied model and complete optimizer config
unchanged, then injects the returned optimizer into Trainer.

The first test uses a real `torch.optim.Muon` returned by a test-local factory.
It verifies model/config identity, optimizer identity and parameter ownership,
configured learning rate/momentum/weight decay, and unchanged model weights
during construction. Adam-specific fields remain available through the exact
config object; Muon does not consume them. The test-local factory uses raw
weight decay to exercise forwarding; it does not define the future combined
Muon/AdamW factory's width-scaled decay or parameter grouping.

## Verified Red

Latest follow-up check: Builder now uses
`optimizer_factory(model.parameters(), config=config.optimizer)`. Running
`./test.sh tests/unit/test_builder.py::test_build_trainer_passes_model_and_complete_config_to_muon_factory --tb=short`
still fails: the factory receives a parameter generator instead of the supplied
model, failing `assert supplied_model is model`. The config keyword is now
correct. Next: pass `model` directly, then migrate legacy SGD test callers and
verify Green. This check changed documentation only; broader tests were not run.

Latest user implementation check: the same targeted command returned exit 1,
**5 failed, 43 passed in 1.55s**. Builder now calls
`optimizer_factory(model.parameters(), lr=config.optimizer)`. The new Muon
factory still rejects `lr`; the four existing SGD cases additionally fail
because an `OptimizerConfig` object is supplied as a numerical learning rate.
The agreed call remains `optimizer_factory(model, config=config.optimizer)`.
No production or test changes were made during this check; old SGD callers
still need test adapters after the Builder contract is implemented. No full
suite was run. The earlier result below is historical.

Reference environment: installed torch **2.15.0+cu132**. Before running the new
test, a standalone real CPU Muon successfully constructed and updated a 2×2
float32 parameter using backward and `step()`, with
`adjust_lr_fn="match_rms_adamw"`. This is an environment check, not a numerical
differential test or evidence of Upwash update behavior.

- Test: `tests/unit/test_builder.py::test_build_trainer_passes_model_and_complete_config_to_muon_factory`.
- Command: `./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py tests/unit/test_training_config.py`.
- Result: exit 1, **1 failed, 47 passed in 2.16s**.
- Expected: Builder calls the factory with the model and `config=config.optimizer`.
- Observed: `TypeError` for unexpected keyword argument `lr` at
  `src/upwash/builder.py:16`; Builder still passes a parameter iterator and `lr`.
- Cause: the agreed factory calling convention is not implemented. The new
  test's factory body and optimizer assertions have not executed via Upwash.
- No production code, dependencies, or test configuration changed. The full
  suite was not rerun.

Next: user implements the agreed Builder call. Existing SGD-based Builder and
integration tests still exercise the previous contract; migrate those callers
to test adapters accepting `(model, *, config)` as part of verifying Green,
preserving their existing numerical expectations. Then agree on the concrete
Muon/AdamW factory, parameter partitioning (including tied embeddings and
norms), explicit model width for `weight_decay / d_model`, and real update
tests. No combined optimizer implementation is provided in this increment.
