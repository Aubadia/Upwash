# 013 — W&B Metric Adapter

## Status

Requested backend on 2026-11-09. API, SDK version, and dependency setup await agreement. No tests or production implementation added.

## Proposed Scope

`WandbLogger(run=run)` wraps a caller-created wandb SDK run targeting W&B or Flock and implements the task 010 `log(metrics, *, step)` interaction. Caller initializes and finalizes the run; one run may span multiple Trainer calls. Translate each event to `run.log(dict(metrics), step=step, commit=True)` after verifying the selected SDK version. Remote adapters should live outside core (proposed `integrations/` package); core depends only on the shared logging interaction.

Confirm task 011 cumulative logging steps first. Do not forward resetting per-call state steps into a single remote run. Optional SDK imports must not prevent using Trainer or MemoryLogger without W&B installed.

## Verification Plan

1. Agree API, lifecycle, exact SDK version, and dependency setup before tests.
2. Adapter unit tests use an injected recording run to check metric/step translation and deliberate commit behavior without network calls.
3. With an approved pinned SDK available, exercise a real offline run and verify persisted scalar history. A fake-only result is not evidence of actual SDK compatibility or remote delivery.
4. Hosted upload is a separate explicitly authorized check; ordinary tests must not create cloud runs.

Research source: [official W&B SDK](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py), inspected unpinned main on 2026-11-09. No exact version selected yet.

Flock source discovery on 2026-11-09 confirms this same SDK path; task 014 verifies the destination rather than implementing a second adapter. See its official-source links and remaining version/testing requirements.
