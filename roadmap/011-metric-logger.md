# 011 — Shared Metric Logging and MemoryLogger

## Status

Current task. The user agreed on batch metric logging on 2026-11-09: prepare a metrics mapping, then call `logger.log(metrics, step=step)`. All four MemoryLogger behaviors are verified Green; the full suite passes. Depends on the completed configuration work in [010](010-training-config.md). Trainer integration and remote-backend proposals remain separate agreements.

## Goal and Agreed Initial Interface

Use one interaction for all metric destinations:

```python
logger.log({"loss": 4.0}, step=1)
```

The contract accepts a mapping of metric names to scalar Python numbers and an explicit integer step. The calling training logic chooses the metrics, converts tensors to detached Python values, and supplies its chosen step. Define a shared contract in `core/contracts.py` when implementing the first interchangeable logging interaction; the concrete typing mechanism remains an implementation decision for the user.

Start with `MemoryLogger` in `core/logger.py`:

```python
logger = MemoryLogger()
logger.log({"loss": 4.0}, step=1)
assert logger.records == [{"step": 1, "metrics": {"loss": 4.0}}]
```

Only MemoryLogger exposes `records`. Preserve insertion order, keep records independent between instances, and copy each incoming metrics mapping so later caller mutation cannot change stored history. Mutation of the exposed records themselves is outside this initial guarantee. Input-type validation beyond these supported inputs needs separate agreement before tests.

## TDD Steps

1. Agree on `log(metrics, *, step)`, MemoryLogger, and the record representation.
2. Write one unit test for multiple metrics in one event and step; confirm the missing API Red without production stubs.
3. User implements the behavior; verify Green.
4. Add successive records, instance independence, and input-mapping isolation in small Red/Green steps.
5. Run relevant regressions and record actual results before moving to [012](012-build-trainer.md).

## Boundaries and Next Action

Callers use the shared interaction without depending on MemoryLogger storage. Task 012 adds optional logger injection and storage in Trainer; actual metric emission stays with the user's training logic. Following the user's 2026-11-12 scope change and merge of tasks 013–014, [013](013-flock-logger.md) covers both FlockLogger through the selected wandb SDK transport and actual Flock delivery verification. W&B service support is outside scope. FlockLogger now initializes its run from caller-supplied project/name/config; completion behavior remains to be agreed in task 013.

CompositeLogger and simultaneous destinations are deferred until explicitly required. They need delivery/error semantics and SDK verification for runs targeting different hosts. File and console backends are also deferred.

## First Red — 2026-11-09

- Test: `tests/unit/core/test_logger.py::test_memory_logger_records_multiple_metrics_at_one_step`.
- Expected: one `log` call records all three scalar metrics (`train/loss`, `train/accuracy`, `optimizer/lr`) in one record at step 1, with the representation above.
- Command: `./test.sh tests/unit/core/test_logger.py` returned exit 2, one collection error in 0.07s: `ImportError: cannot import name 'MemoryLogger' from 'upwash.core.logger'`.
- Cause: the production logger module is empty. This is the intended initial API-availability Red; the record assertion has not executed. No production stub was added.
- The preceding full-suite baseline was `./test.sh` → `38 passed in 4.07s`. The full suite has not been rerun after adding this Red test.

## Initial Green and Mapping Isolation Red — 2026-11-09

- User implemented MemoryLogger. `./test.sh` returned exit 0, `39 passed in 2.26s`, verifying the initial record assertion and all prior tests.
- Added tests for successive record ordering and independent logger instances. `./test.sh tests/unit/core/test_logger.py` returned `3 passed in 0.01s` on their first run; no Red is claimed for these two already-supported behaviors.
- Added `test_memory_logger_preserves_metrics_when_caller_changes_input`. After logging loss `4.0` at step 1, the caller changes loss to `2.56` and adds accuracy `0.75`. Expected stored history remains exactly the original loss record.
- `./test.sh tests/unit/core/test_logger.py` returned exit 1, `1 failed, 3 passed in 0.02s`. The stored record instead contains the changed loss and added accuracy because it holds the original mapping by reference. This is the intended isolation Red.
- The full suite has not been rerun after adding these tests; `39 passed` is the preceding checkpoint with the first logger test only.

Next action: user snapshots the incoming scalar metrics mapping when recording an event, then agent verifies Green and regressions. A shallow mapping copy is sufficient for the agreed scalar-only values. The current implementation accepts positional step too; the agreed keyword call is tested, while strict keyword-only rejection is not tested. No agent production edits, dependency changes, or test configuration changes were made.

## Final Green — 2026-11-09

The user now snapshots incoming metrics with `copy.deepcopy`. `./test.sh` returned exit 0, `42 passed in 2.24s`, covering all four logger tests and existing regressions. Record contents, ordering, instance independence, and isolation from caller mapping mutations are verified. The agent changed no production code, dependencies, or test configuration.

For scalar-only values, a shallow mapping copy would suffice; simplifying the deep copy is an optional user refactor. Strict keyword-only step enforcement is not tested. `core/contracts.py` remains empty: shared typing is not implemented or verified, and can be introduced when the next interchangeable logger interaction needs it.

This Green supersedes the preceding mapping-isolation handoff. Next action: proceed directly to task 012 Builder. Its interface is now agreed, including optional logger injection and storage in Trainer. On 2026-11-09 the user removed the automatic training-loss logging task; automatic loss logging and callbacks are not part of the route.
