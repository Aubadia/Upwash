# 013 — Verify the Flock Logging Destination

## Status

Planned after [012](012-wandb-logger.md). Official source documentation confirms the wandb SDK path; executable server compatibility and delivery remain unverified. No separate FlockLogger is planned. Exact SDK/server versions and an integration environment await agreement.

## Goal

Verify task 012's WandbLogger with a run targeting Flock. The caller configures and owns the run; the caller chooses metrics and steps, while Builder only passes the logger through and Trainer stores it without automatic metric emission. Reuse adapter unit tests rather than duplicating an adapter for a second host.

The Flock guide documents process-specific `WANDB_BASE_URL=https://flock-api.cerulean.works` and `WANDB_API_KEY`. Its CLI token is distinct from the wandb key. Configuration examples are not instructions to modify this machine's credentials or SDK defaults.

## Evidence and Version Selection

Official source guides retrieved on 2026-11-09 confirm ordinary `wandb.init`, `run.log`, completion by context manager or `run.finish`, explicit steps, and offline logging with later sync. The compatibility guide reports testing with SDK 0.31 without an exact patch version. Artifacts, sweeps, reports, and `wandb.Api()` are unsupported.

Sources:

- [Experiment guide](https://github.com/cerulean-works/flock/blob/main/docs/source/app/log-runs.md)
- [Logging behavior](https://github.com/cerulean-works/flock/blob/main/docs/source/app/logging-behavior.md)
- [Compatibility](https://github.com/cerulean-works/flock/blob/main/docs/source/app/wandb-compatibility.md)
- [Credentials](https://github.com/cerulean-works/flock/blob/main/docs/source/cli/sign-in.md)

These guides were retrieved from main; the repository API reported HEAD `9055b0d947e2622a301f7dc05dc5322c4c2cd58a` during discovery. This is a research snapshot, not a verified deployed-server version. Record an exact SDK version and a verified server revision before execution. Hosted documentation initially returned HTTP 403; source access resolved that discovery issue.

## Verification Steps

1. Agree on pinned SDK/server versions and a controlled local server or explicitly authorized hosted destination. Discuss required dependency/configuration changes first.
2. Log known scalar values and increasing steps through the actual shared adapter into a real Flock run.
3. Finalize the caller-owned run, then inspect persisted values, steps, and completed status through Flock's documented query API; do not use unsupported `wandb.Api()`.
4. Report SDK/server versions, command, destination type, observed values, and cleanup ownership. Distinguish delivery failures from adapter defects or environment blockers.

Task 012's offline result cannot establish Flock delivery. Simultaneous W&B/Flock runs in one process need separate SDK validation if later requested. No service has been started, credential changed, or metric uploaded as part of planning.

Next action after 012: agree on and prepare the integration environment before attempting delivery verification.
