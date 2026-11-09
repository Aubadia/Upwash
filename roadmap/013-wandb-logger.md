# 013 — Shared wandb SDK Adapter

## Status

Current task after [012 Builder](012-build-trainer.md), verified complete with `./test.sh` → `46 passed in 2.11s` on 2026-11-09. Depends on the shared metric interaction from [011](011-metric-logger.md). W&B and Flock support are requested. Adapter API, exact SDK version, and dependency setup await agreement. No tests or implementation added.

## Goal and Proposed Interface

```python
logger = WandbLogger(run=run)
logger.log({"loss": 4.0}, step=1)
```

Wrap a caller-created wandb SDK run. Translate each event to `run.log(dict(metrics), step=step, commit=True)`, subject to verification against the chosen SDK version. Each call represents one complete metric event. Do not retain the whole history in the adapter.

The same adapter serves a run configured for W&B or Flock. Host selection, authentication, run initialization, and completion belong to the caller. The calling training logic chooses metrics, logging frequency, and valid increasing run steps. Trainer and Builder do not own logging or remote run lifecycle.

Place the remote adapter outside core (proposed `integrations/` package; exact module path awaits agreement). The adapter follows the shared logging interaction; Trainer stores the supplied logger through the shared interaction without importing the remote adapter or SDK. Using Trainer or MemoryLogger must not require importing the optional wandb SDK.

## Reference and Dependency Decisions

The official [Run implementation](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py), inspected on 2026-11-09 from unpinned main, documents explicit-step and commit behavior. Committed history cannot reset to earlier steps. This is research evidence, not a pinned compatibility oracle.

Before executable SDK tests, select and record the exact wandb version and relevant environment versions. Flock documentation reports SDK 0.31 compatibility without an exact patch version; see [014](014-flock-logger.md). Discuss optional dependency/test configuration changes before making them. No dependency installation is authorized by this plan alone.

## TDD and Verification Steps

1. Agree on the adapter interface, caller ownership, SDK version, and dependency setup.
2. With an injected recording run, confirm Red for one metric/step translation, including commit behavior and preservation of the caller's mapping. User implements the adapter; verify Green.
3. Exercise the real pinned SDK in offline mode and inspect persisted metric values and increasing steps. Offline verification establishes SDK serialization, not hosted delivery.
4. Verify caller-supplied metric events over multiple adapter calls and that ordinary core imports work without the optional SDK.
5. Record results and limitations before destination-specific verification in task 014.

Hosted delivery checks require an agreed destination and explicit upload authorization. Ordinary unit tests use no network. Concurrent multi-host runs, resume, media, artifacts, sweeps, and retry policies are deferred.
