# 014 — Verify the Flock Logging Destination

## Status

Documentation discovery resolved on 2026-11-09 using the repository supplied by the user. No adapter tests or production implementation added. Exact SDK and server versions remain to be selected before integration testing.

## Findings and Revised Scope

Official guides confirm that Flock uses the wandb SDK: `wandb.init`, `run.log`, context-managed completion or `run.finish`. A separate `FlockLogger` is unnecessary for scalar metrics; reuse task 013's `WandbLogger` with a caller-configured run.

For process-specific destination selection, the guide documents `WANDB_BASE_URL=https://flock-api.cerulean.works` and `WANDB_API_KEY`. CLI credentials are distinct from the wandb key. Caller owns run creation and completion; Trainer does not change login settings. Offline logging and later `wandb sync` are supported.

The compatibility guide states testing with SDK 0.31, without an exact patch version. Artifacts, sweeps, reports, and `wandb.Api()` are unsupported. This is documentation evidence, not an executed compatibility test.

Sources retrieved from main on 2026-11-09:

- [Experiment guide](https://github.com/cerulean-works/flock/blob/main/docs/source/app/log-runs.md)
- [Logging behavior](https://github.com/cerulean-works/flock/blob/main/docs/source/app/logging-behavior.md)
- [Compatibility](https://github.com/cerulean-works/flock/blob/main/docs/source/app/wandb-compatibility.md)
- [Credentials](https://github.com/cerulean-works/flock/blob/main/docs/source/cli/sign-in.md)

Repository API reported HEAD `9055b0d947e2622a301f7dc05dc5322c4c2cd58a` during discovery. Pin and verify a server revision and exact SDK version before integration testing. The previous hosted-documentation HTTP 403 blocker is resolved by these source guides.

## Verification Plan

1. Agree shared logger API, lifecycle, exact SDK version, and dependency setup.
2. Reuse task 013 adapter tests for metric/step translation. Do not duplicate a second adapter.
3. Real SDK offline checks verify serialization, not Flock delivery.
4. Agree a controlled local Flock server integration or explicitly authorized hosted smoke check. Verify metrics, steps, and completed run state using Flock's documented query API rather than unsupported `wandb.Api()`.

Support for either destination does not establish simultaneous multi-host runs in one process; that requires separate SDK validation if requested. No SDK installed, credentials changed, service started, or metrics uploaded. API proposals remain pending user agreement.
