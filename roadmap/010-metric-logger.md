# 010 — Minimal Metric Logger

## Status and Dependency

Current task after [009](009-training-config.md), verified Green on 2026-11-08 with `./test.sh`: `25 passed in 2.02s`. Logger module remains empty. API below is provisional and requires agreement before tests; no Logger tests written yet.

## Proposed Behavior

Scope expanded by the user on 2026-11-09: support W&B and Flock, in addition to in-memory observation. Revised proposal: define the shared `log(metrics, *, step)` interaction, with `MemoryLogger` as the first concrete implementation. Only `MemoryLogger` exposes `records`; remote adapters need not retain the whole history. A call such as `logger.log({"loss": 4.0}, step=1)` appends a memory record represented as `{"step": 1, "metrics": {"loss": 4.0}}`. Metrics are scalar Python numbers; Trainer will perform tensor detachment and scalar conversion in task 011.

Start with in-memory storage so users and tests can inspect training metrics. Preserve insertion order and copy the incoming metrics mapping so later caller mutation does not rewrite a stored event. Different logger instances have independent records. External mutation of exposed records is outside this initial contract. Task 013 proposes one `WandbLogger` wrapping a supplied run; task 014 verifies it against Flock. Flock source guides confirm wandb SDK support, so a separate `FlockLogger` is unnecessary for scalar metrics. Only the memory backend needs observable records; remote adapters need not retain history. If simultaneous destinations are agreed, add a `CompositeLogger` forwarding each event to all supplied loggers, with equivalent metric contents and without sharing mutable input between destinations. Error propagation and partial-delivery behavior must be agreed before fan-out tests.

## TDD Sequence

1. Agree on the logger interface and record representation.
2. Test one recorded metric and step, confirm Red, then hand implementation to the user.
3. Cover successive records, instance independence, and input-mapping isolation in small steps.

Future Trainer integration should depend on the `log` interaction, allowing a recording test double or another backend without coupling core to this storage implementation. Introduce a shared contract only when that interaction requires it.

No tests run for this feature; no implementation or Red claimed.

## Backend Research — 2026-11-09

- W&B SDK source confirms `Run.log(data, step=None, commit=None)` and increasing history steps; explicit-step calls require deliberate commit behavior. Source: [official SDK](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py). The inspected source is unpinned `main`, research only. Select and record an exact SDK version before executable adapter tests.
- Both supplied Flock URLs returned HTTP 403 when fetched directly: [application](https://flock.cerulean.works/), [guide](https://docs.cerulean.works/flock/main/guide/). No package name, SDK signature, W&B compatibility, or lifecycle behavior has been verified. Need accessible official documentation/source or a user-provided SDK example before specifying that adapter.
- Proposed initial adapters wrap caller-created runs/clients. Caller owns initialization and finalization, allowing one run to span repeated `train()` calls. Trainer does not initialize or finish remote runs. Builder initially accepts the supplied logger unchanged.
- No SDK installed, account accessed, run created, or metrics uploaded. Dependency additions and optional-dependency configuration require discussion per AGENTS.md. Revised API still awaits agreement.

## Flock Discovery Update — 2026-11-09

The user supplied [the Flock repository](https://github.com/cerulean-works/flock). Official guides resolve the earlier documentation-access blocker: Flock accepts the wandb SDK, supports run completion and offline sync. See task 014 for sources. Revised API proposal: shared `log(metrics, *, step)`, concrete `MemoryLogger` and `WandbLogger`, and optional `CompositeLogger`. Rename the earlier proposed concrete `Logger` with records to `MemoryLogger`; the shared interaction need not expose records. No new API agreement is inferred from the repository link.

Caller owns each run and selects its destination before injection. Simultaneous W&B/Flock runs in one process require SDK validation before promising that capability. Dependency setup, exact versions, fan-out failures, and step/lifecycle proposals still need agreement. No tests run for this documentation update.
