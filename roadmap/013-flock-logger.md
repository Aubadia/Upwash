# 013 — Flock Logger and Delivery Verification

## Status and Scope

2026-11-13 follow-up: [019 — Training Loop Logging](019-training-loop-logging.md)
now covers computing/emitting metrics from Trainer with opt-in `logging_steps`
configured through TrainingConfig/Builder. The older no-automatic-emission scope
below describes task 013's logger adapter; default logging remains disabled in
019's initial design. Real remote delivery is still unverified.

Merged former tasks 013 (run adapter) and 014 (Flock delivery verification) at the user's request on 2026-11-12. Depends on the shared logging interaction from [011](011-metric-logger.md); [012](012-build-trainer.md) is complete. Flock is the only requested remote destination. The public class name is now agreed as FlockLogger, implemented by the user in `src/upwash/core/logger/flock.py` and re-exported through `src/upwash/core/logger/__init__.py`.

The constructor, seven-metric forwarding, and exit-callback unit tests are Green. Actual SDK serialization, executable server compatibility, and Flock delivery remain unverified. Completing the unit tests alone does not complete this task: actual Flock delivery must also be checked. W&B service support is outside scope.

## Agreed Interface and Ownership

```python
from upwash.core.logger import FlockLogger

logger = FlockLogger(
    project="lab6-predictable-scaling", name="tiny-training", config={"lr": 0.1, "batch_size": 4}
)
logger.log({
    "train/loss": 4.0,
    "train/grad_norm": 1.5,
    "train/learning_rate": 0.1,
    "train/tokens_per_second": 2048.0,
    "train/total_tokens_seen": 8192,
    "eval/loss": 1.0,
    "eval/perplexity": 2.718281828459045,
}, step=7)
```

- The constructor is `FlockLogger(project, name, config=None)`, following the user's correction and source interface. FlockLogger creates its run through `wandb.init(project=project, name=name, config=config)`. The tested scope currently supplies an explicit nonempty config mapping; omitted/empty config normalization has not been tested. The caller supplies these experiment settings and configures the Flock host and credentials. Trainer and Builder do not initialize runs. The user has now implemented finish() with an atexit fallback. The recording-run regression checks one registered callback can finish the run; repeated finish, failure recovery, and logging after completion are not yet tested.
- `log(metrics, *, step)` forwards one complete event as `run.log(dict(metrics), step=step, commit=True)`. It preserves metric values and the supplied step, does not modify the caller's mapping, and forwards an independent mapping. The adapter does not retain the entire history.
- The caller selects detached scalar metrics, logging frequency, and valid increasing SDK steps. `state.global_step` remains per-call; no automatic Trainer metric emission or cumulative logging counter is introduced.
- Builder passes through an optional logger; Trainer stores it without depending on the concrete implementation. Ordinary core/MemoryLogger imports must remain independent of loading the remote SDK.
- Resume, shared runs, CompositeLogger, multi-host delivery, media, artifacts, sweeps, and retry policy are deferred.

## Required Test Project and Metrics

The user fixed the Flock test project to `lab6-predictable-scaling`. Use it both in mocked constructor tests and in the eventual real Flock delivery scenario. Run names remain supplied through `name`; they are not fixed by this requirement.

| Metric | Required meaning |
| --- | --- |
| `train/loss` | Average loss over the `logging_steps` interval. |
| `train/grad_norm` | Gradient norm before clipping. |
| `train/learning_rate` | Training learning rate; log parameter groups separately when they use different rates. |
| `train/tokens_per_second` | Training throughput at the logging step. |
| `train/total_tokens_seen` | Cumulative non-padding training tokens processed. |
| `eval/loss` | Token-averaged cross-entropy on the designated development set. |
| `eval/perplexity` | Perplexity calculated as exp(eval/loss). |

The current unit event contains exactly these seven keys and known scalar fixture values, including integer cumulative tokens and eval/loss 1.0 paired with perplexity exp(1.0). It verifies transport of supplied values; it does not verify loss aggregation, gradient measurement, timing, token counting, or evaluation computation. Those values remain the responsibility of training/evaluation code, and no automatic Trainer metric emission is introduced. The logger remains generic; no filtering or restriction to these keys is required. The current fixture uses one parameter group; the names for distinct group learning rates still need agreement before tests for that case.

## Current Source and Test State

The user has implemented `FlockLogger(project, name, config=None)` in `core/logger/flock.py`, forwarding settings to `wandb.init` with `config or {}`. The module now imports successfully. Its `log` method currently calls `self.run.log(metrics, step=step)`; explicit commit and mapping isolation are not yet verified. The user also added finish() and registered it with atexit. FlockLogger is now re-exported by `core/logger/__init__.py`. MemoryLogger remains re-exported by the package. The agent changed tests and documentation only.

`tests/unit/core/test_flock_logger.py` contains three cases:

- Exit callback regression: capture the callback registered by the real constructor and invoke it explicitly; RecordingRun.finish is called once. This passes after correcting the test double.

- Constructor forwarding: project `lab6-predictable-scaling`, supplied name, and explicit nonempty config reach `wandb.init` unchanged. This passes.
- Metric-event forwarding: construct through the agreed package import with that project and send the seven-key payload at step 7 with `commit=True`, preserving input and mapping independence. This stops at the missing package export; forwarding assertions have not executed.

All three cases patch `wandb.init` to return a RecordingRun. Each also intercepts atexit registration so callbacks do not leak into process exit; the exit regression explicitly invokes only its captured callback. Pytest restores the initializer after each case. No real SDK run is created and nothing is uploaded. The constructor test uses the defining module to test initialization independently of the missing re-export.

## Current Verification — 2026-11-12

Latest after the user's mapping-copy implementation: `./test.sh` → exit 0; `65 passed in 1.95s`. Constructor, seven-metric forwarding (values/step/commit/input isolation), and exit-callback regression are Green. This supersedes the earlier Red records below. All Flock tests patch wandb.init and use RecordingRun; no real SDK run or upload occurs, even with project lab6-predictable-scaling. Source search found no separate real Flock logging scenario under src/scripts/tests. The requested dashboard URL could not be inspected with the web tool, so no claim is made about existing server runs. Next: prepare actual Flock delivery verification with the selected organization/project, host/key setup, a distinct run name, and explicit completion; obtain authorization for the concrete hosted upload before execution. No credentials were read or changed and no network logging occurred.

Latest after explicit commit forwarding: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.23s`. The metric-event test now verifies all seven metric values, step 7, commit=True, and unchanged caller input immediately after log. Its final assertion fails: clearing the caller's metrics also clears RecordingRun's received mapping (`{}` instead of seven values). This confirms the intended mapping-isolation Red. Next: user forwards a separate dict to the SDK, preserving the current step/commit behavior. The full suite was not rerun. Earlier commit/export failures below are historical. The user also added an optional commit argument; commit=False behavior has not been agreed or tested.

Latest after the user's package export: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.25s`. Constructor and exit callback pass. Metric forwarding now reaches `flock.py:20` and raises `TypeError: RecordingRun.log() missing 1 required keyword-only argument: 'commit'` because production calls `run.log(metrics, step=step)`. This is the intended forwarding Red for the agreed explicit commit behavior. Input-isolation assertions have not executed; the source still passes the caller's mapping directly. Next: user forwards an independent mapping and `commit=True`, then rerun the targeted tests. The full suite was not rerun for this check. Earlier missing-export outcomes below are historical.

Latest after test-double cleanup: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.24s`. `./test.sh` → exit 1; `1 failed, 64 passed in 1.80s`. Constructor and exit-callback tests pass; the sole failure is the missing FlockLogger package export. No ignored atexit exception appeared on either run. Earlier table entries below predate this correction.

### RecordingRun finish Regression — 2026-11-12

- Reproduced the user's error with the existing logger tests: the constructor test passed, but actual process exit raised AttributeError because RecordingRun lacked finish(). This was a test-support defect, separate from the package-export failure.
- Added `test_flock_logger_exit_callback_finishes_the_recording_run`. Before the fix, `./test.sh tests/unit/core/test_flock_logger.py::test_flock_logger_exit_callback_finishes_the_recording_run` → `1 failed in 0.20s`, reproducing that exact AttributeError inside the captured callback.
- Added finish() and a call counter to RecordingRun, then isolated atexit registration in all three tests. No production code, exception suppression, SDK settings, or dependencies were changed.
- The targeted and full-suite outcomes above verify the cleanup regression Green while preserving the separate missing-export failure. Metric assertions still have not executed; no remote run was created or uploaded.

### Earlier Verification

| Command | Result | Interpretation |
| --- | --- | --- |
| `./test.sh tests/unit/core/test_flock_logger.py tests/unit/core/test_logger.py` (fixed project and seven metrics) | Exit 1; `1 failed, 5 passed in 0.28s`. | Constructor and four MemoryLogger tests pass. Sole failure: missing FlockLogger package export. Metric assertions have not executed. |
| `./test.sh tests/unit/core/test_flock_logger.py` (earlier unfinished log body) | Exit 1; `2 failed in 0.02s`. | Historical IndentationError, now resolved; not the intended behavioral Red. |
| Same targeted command, before the unfinished log declaration | Exit 1; `1 failed, 1 passed in 1.35s`. | Historical constructor Green; metric test stopped at missing export. |
| `./test.sh` (earlier project/payload) | Exit 1; `1 failed, 63 passed in 2.04s`. | Earlier regression evidence only. Full suite not rerun after the project/metric fixture update. |

Next: user re-exports FlockLogger from `core/logger/__init__.py` and completes the agreed metric forwarding in `flock.py`; rerun the targeted tests. The current source forwards step but does not explicitly commit or copy the mapping; verify those behaviors once the test reaches the method. finish() is now implemented with an atexit fallback; only the single captured-callback path is tested. Default-config behavior, multi-event/optional-import behavior, metric calculation semantics, real SDK execution, and actual Flock delivery remain unverified.

## SDK and Documentation Evidence

The inspected environment contains Python 3.14.6, pytest 9.2.0, torch 2.15.0, numpy 2.5.4, wandb 0.31.0, huggingface_hub 2.4.0, and pyarrow 26.0.1. The user's `pyproject.toml` declares `wandb>=0.31.0`; the installed version is the exact recorded SDK target for the first executable check, but the declaration itself is not a pin. The agent changed no dependencies, lockfile, installation, or test configuration. Discuss any further dependency/configuration changes before making them.

Official hosted documentation read successfully on 2026-11-12:

- [Guide](https://docs.cerulean.works/flock/main/guide/)
- [Track an experiment](https://docs.cerulean.works/flock/main/guide/app/log-runs/)
- [Sign in and manage access](https://docs.cerulean.works/flock/main/guide/cli/sign-in/)
- [Logging behavior](https://docs.cerulean.works/flock/main/reference/app/logging-behavior/)
- [wandb compatibility](https://docs.cerulean.works/flock/main/reference/app/wandb-compatibility/)

The guide uses caller-owned `wandb.init` runs, `run.log`, and completion through a context manager or `run.finish`. For per-process configuration it documents `WANDB_BASE_URL=https://flock-api.cerulean.works` and `WANDB_API_KEY`. A CLI `FLOCK_TOKEN` cannot replace the SDK key. No login, credential setup, or SDK defaults were changed during this work.

The compatibility reference documents SDK 0.31 without an exact patch version. Its source links identify documentation revision `710e75a857969be3fefef67c50b3df016f15ef15`, not a deployed server version. Initial source research on 2026-11-09 recorded repository HEAD `9055b0d947e2622a301f7dc05dc5322c4c2cd58a`; that also was not a deployed-server compatibility result. Hosted documentation access initially failed but succeeded using the user's guide URL.

The [wandb 0.31.0 Run source](https://github.com/wandb/wandb/blob/v0.31.0/wandb/sdk/wandb_run.py) documents explicit-step commit behavior: supplying a step defaults to an uncommitted event, so the agreed adapter passes `commit=True`. Source inspection is not SDK execution. Flock's query API should be used for automated delivery inspection; `wandb.Api()` is unsupported.

## Remaining TDD and Delivery Steps

1. Constructor forwarding with project `lab6-predictable-scaling` is Green. Complete the package export and seven-metric event forwarding, then verify the metric test Green.
2. Verify multiple caller-supplied events and ordinary core/MemoryLogger imports without the remote SDK. Add only tests needed for these behaviors.
3. Use the implemented finish() for explicit completion, then select and record the exact SDK/server versions and a controlled local server or explicitly authorized hosted Flock destination. Use project `lab6-predictable-scaling`; confirm organization, run name, verification method, and cleanup ownership before creating a run. No destination has been selected for execution yet.
4. Through FlockLogger's project/name/config constructor and the real SDK, create a fresh Flock run in `lab6-predictable-scaling` and log known values for the seven required metrics at increasing steps. Finalize the run using the agreed completion API, then check persisted values, steps, and completed state using the Flock dashboard or documented query API. Never use `wandb.Api()` for this check.
5. Record the command, versions, destination type, observed values/state, and any delivery or environment failures. Unit-test Green and offline serialization do not establish delivery. An offline SDK check is an optional diagnostic if needed.

No service was started, metric uploaded, or remote run created during planning. Ordinary unit tests use no network. Hosted delivery still requires the agreed destination and explicit upload authorization.

## Historical Adapter Iterations

Before the constructor correction, `./test.sh tests/unit/core/test_flock_logger.py` returned `1 failed in 0.02s`, and `./test.sh` returned `1 failed, 62 passed in 1.67s`, both for the missing FlockLogger import. These results predate the user's constructor implementation and the new initialization test.

The following records predate the agreed FlockLogger name and the merge of tasks 013–014. Old class names, filenames, commands, and next-action notes describe those historical states; the current interface and next action above supersede them.

### Initial Session Verification — 2026-11-12 (Historical)

- `main` includes PR #3 at `7e60792`; no remote adapter source or tests were found.
- `./test.sh tests/unit/test_training_config.py tests/unit/core/test_logger.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → `14 passed in 2.95s` (exit 0).
- `./test.sh` → eight collection errors (exit 2): local `.venv` lacks `huggingface_hub` and `pyarrow`. This is an environment blocker, not the intended Red for task 013. Dependencies and test configuration were not changed.
- This initial inspection preceded the API acceptance and first Red recorded below.

### Earlier SDK and Dependency Research


The official [Run implementation](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py), inspected on 2026-11-09 from unpinned main, documents explicit-step and commit behavior. Committed history cannot reset to earlier steps. This is research evidence, not a pinned compatibility oracle.

Proposed real SDK target: `wandb==0.31.0`. The [release page](https://pypi.org/project/wandb/0.31.0/) lists Python 3.14 support. The [version-tagged Run source](https://github.com/wandb/wandb/blob/v0.31.0/wandb/sdk/wandb_run.py), inspected on 2026-11-12, accepts `data`, `step`, and `commit`; with an explicit step, commit defaults to false. Explicit `commit=True` finalizes the metric event. This is source inspection, not execution of the SDK.

Initial environment snapshot: Python 3.14.6, pytest 9.2.0, torch 2.15.0, numpy 2.5.4; wandb was absent. The later installed SDK version and existing dependency declaration are recorded in SDK and Documentation Evidence above, superseding the earlier optional-extra proposal. The recording-run unit test needs no SDK. Further dependency/configuration changes still require discussion before changes. Flock's previously reported 0.31 compatibility does not verify this exact SDK/server pair; the former task 014's delivery steps are now included in this task.

### First Verified Red — 2026-11-12

- Test: `tests/unit/integrations/test_wandb_logger.py::test_wandb_logger_forwards_one_complete_metric_event_with_an_independent_mapping`.
- Command: `./test.sh tests/unit/integrations/test_wandb_logger.py` → exit 1; `1 failed in 0.01s`.
- Expected: one call to the supplied run containing both scalar metrics, `step=7`, and `commit=True`; caller input remains unchanged, and clearing it later does not change the received mapping.
- Observed: `ModuleNotFoundError: No module named 'upwash.integrations'` at the adapter import inside the test.
- Cause: missing Upwash API. No production stub was created. The assertions have not yet executed; no SDK compatibility or hosted delivery is claimed.
- This initial import path was superseded by the user's placement decision below.

## Placement Correction — 2026-11-12

- Agreed public import: `from upwash.core.logger import WandbLogger`; implementation belongs in the existing `src/upwash/core/logger.py`.
- Moved the unchanged behavioral test to `tests/unit/core/test_wandb_logger.py` and updated its import. Production source and dependencies were not changed.
- Verification: `./test.sh tests/unit/core/test_wandb_logger.py tests/unit/core/test_logger.py tests/unit/test_training_config.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → exit 1; `1 failed, 14 passed in 1.45s`.
- Intended Red: `ImportError: cannot import name 'WandbLogger' from 'upwash.core.logger'`. The module exists; the class is missing. Event assertions have not executed yet. Existing MemoryLogger, Config, and Builder tests remain Green; the full suite was not rerun because its previously recorded environment blocker remains unresolved.
- The later source-package conversion below supersedes this single-file placement.

## Source Package Conversion — 2026-11-12

- The user explicitly requested creating and moving the source folders/files. The agent converted `core/logger.py` into `core/logger/`, moved MemoryLogger unchanged to `memory.py`, and re-exported it through `__init__.py`.
- Created `wandb.py` with a module docstring only. No WandbLogger class, methods, or behavior were added; the existing Red test remains the implementation handoff.
- Verification: `./test.sh tests/unit/core/test_wandb_logger.py tests/unit/core/test_logger.py tests/unit/test_training_config.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → exit 1; `1 failed, 14 passed in 1.31s`. The expected Red remains the missing WandbLogger import, now from the package's `__init__.py`; all selected existing regression cases pass. MemoryLogger source also matches its pre-move Git version byte-for-byte. The full suite was not rerun; its previously recorded data-dependency blocker remains unresolved.
- Next: user implements WandbLogger in `core/logger/wandb.py` and re-exports it through `core/logger/__init__.py`, then verify Green. Dependency/test configuration remains unchanged.

