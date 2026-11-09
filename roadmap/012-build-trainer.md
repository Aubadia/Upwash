# 012 — Build a Configured Trainer

## Status

Current task after [010](010-training-config.md) and [011](011-metric-logger.md). On 2026-11-09 the user agreed on the construction interface below, including optional logger injection and storage in Trainer. Automatic training-loss logging remains removed. Complete: construction, optional logger, real optimizer wiring, and caller-owned logging behavior verified Green. No agent production edits.

## Goal and Agreed Interface

```python
from upwash.builder import build_trainer

trainer = build_trainer(
    config=config,
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=torch.optim.SGD,
    logger=logger,
)
state = trainer.train(loader)
```

Start with a function. Inputs are the composed TrainingConfig, an existing model and loss function, an explicit optimizer factory, and optional `logger=None`. Construct the optimizer with the supplied model's parameters and `lr=config.optimizer.lr`, then return a wired Trainer without starting training. Pass the logger unchanged to Trainer, whose constructor also accepts optional `logger=None` and stores it as public `trainer.logger`.

Config contains settings; the builder constructs the optimizer; Trainer receives the runtime optimizer. Core does not import the builder or concrete optimizer implementations. The supplied factory must accept parameters and keyword `lr`; additional optimizer options are deferred.

## Ownership

The caller creates the model, loss function, data loader, and logger. Builder injects the supplied logger; Trainer retains that exact object for the user's training logic. The caller decides metric names, logging frequency, steps, and actual log calls. Construction and the existing `train()` flow do not automatically invoke the logger. No callback or replacement training-loop extension is part of this task. Backend selection, credentials, and remote run lifecycle stay with the caller.

## TDD Steps

1. Write the first test against the agreed construction interface; confirm Red before user implementation.
2. Verify construction leaves model parameters unchanged and retains the exact supplied logger as `trainer.logger`. Check that construction emits no log events.
3. Exercise the built Trainer with real CPU SGD: initial weight zero, input `1`, target `2`, mean MSE, configured `lr=0.1`; expect weight approximately `0.4` and returned step `1`.
4. Use `lr=0.05` and expect weight approximately `0.2`, detecting ignored configuration or incorrect optimizer wiring.
5. Verify omitted logger gives `trainer.logger is None` and existing direct Trainer construction remains usable. With a supplied recording logger, the existing training flow emits no automatic metrics.
6. Run relevant regressions and the full suite.

Each behavior gets a small Red/Green step. Real updates must affect the supplied model. Model factories, loader construction, scheduler construction, registries, serialization, and complete run orchestration are deferred.

## First Red — 2026-11-09

- Test: `tests/unit/test_builder.py::test_build_trainer_constructs_configured_optimizer_and_preserves_supplied_objects`.
- Expected: return a Trainer holding the supplied model, loss function, and exact logger, with a real SGD optimizer configured at `lr=0.1`. Construction leaves model weights unchanged and creates no metric records.
- Command: `./test.sh tests/unit/test_builder.py` returned exit 2, one collection error in 2.58s: `ImportError: cannot import name 'build_trainer' from 'upwash.builder'`.
- Cause: the builder module is empty. This is the intended initial API-availability Red; construction assertions have not executed. No production stub was added.
- Last full-suite Green was `./test.sh` → `42 passed in 2.24s`, before this test. The full suite has not been rerun after adding the Red test.

Next action: user implements `build_trainer` and optional Trainer logger storage, then agent verifies Green. Real parameter-update wiring, a second learning rate, omitted logger, and absence of automatic metrics during training remain subsequent steps. No dependencies or test configuration changed.

## Implementation Check — 2026-11-09

- User implemented Builder with the supplied factory and `config.optimizer.to_dict()`, plus Trainer logger storage.
- `./test.sh tests/unit/test_builder.py` returned exit 0, `1 passed in 2.03s`. The construction assertions now execute and pass.
- `./test.sh` returned exit 1, `10 failed, 33 passed in 2.12s`. All ten failures are existing direct Trainer construction cases: `TypeError: Trainer.__init__() missing 1 required positional argument: 'logger'`.
- Cause: `logger: None` is a type annotation, not a default. Trainer and build_trainer both currently require the argument, contrary to the agreed optional `logger=None` API. The existing regression tests correctly omit it and should not be changed to hide this regression.
- The initial Builder test always supplies a logger, so it does not cover omission. Real update wiring, a second learning rate, omitted Builder logger, and no automatic training metrics remain later tests.

Current next action: user gives logger a default of None in both constructors/functions, then agent reruns relevant tests and full regressions. No agent production edits, dependency changes, or test configuration changes were made. Task 012 is not complete.

## Trainer Default Green and Builder Default Red — 2026-11-09

- User changed Trainer's logger default to None. Before the new test, `./test.sh` returned exit 0, `43 passed in 2.15s`; the ten direct-Trainer regressions are resolved.
- Builder still requires logger. Added `tests/unit/test_builder.py::test_build_trainer_allows_omitting_logger`, calling the agreed Builder API without logger and expecting `trainer.logger is None`.
- `./test.sh tests/unit/test_builder.py` returned exit 1, `1 failed, 1 passed in 1.90s`. Observed `TypeError: build_trainer() missing 1 required positional argument: 'logger'`. This is the intended optional-argument Red.
- The full suite was not rerun after adding this test; `43 passed` describes the preceding checkpoint.

Current next action: user adds the agreed `logger=None` default to build_trainer; agent then verifies Green. Real training with two learning rates and no automatic metrics remains a subsequent integration step. No agent production edits, dependency changes, or test configuration changes were made.

## Final Green — 2026-11-09

- User added `logger=None` to Builder. Before new integration tests, `./test.sh` returned exit 0, `44 passed in 2.14s`, including the omitted-logger test.
- Added `tests/integration/test_built_trainer.py`, using real CPU float32 gradients and SGD at learning rates 0.1 and 0.05. Starting from weight zero, input 1, and target 2, the supplied model reaches weights approximately 0.4 and 0.2 respectively. Returned step is 1 and the supplied MemoryLogger remains empty during training.
- `./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py` returned exit 0, `4 passed in 2.05s`.
- `./test.sh` returned exit 0, `46 passed in 2.11s`.
- New integration cases passed on their first run; no Red is claimed for already-implemented behavior. They verify configured learning rates reach the correct model parameters and that training does not automatically emit metrics.
- No agent production edits, dependency changes, or test configuration changes were made.

This Green supersedes earlier handoffs. Next action: agree on task 013's wandb adapter interface, exact SDK version, and dependency/test setup before its first Red. No remote SDK compatibility or hosted delivery is established by Builder tests.
