# Implementation Progress

Last updated: 2026-11-13 (Asia/Taipei).

## Resume Here

Current priority: [019 — Training Loop Logging](019-training-loop-logging.md).
Token counting is implemented in Builder and Trainer with defaults
`count_tokens=False`, `pad_token_id=None`. Enabled counting sums input tokens
across training calls, excludes configured padding ids and evaluation, and
continues while logging is disabled. Both event paths emit Python integers.
Added coverage for padding ids 0/1, all-padding inputs, explicit/default
disabled counting, repeated calls, evaluation exclusion, and logger omission.
Targeted logging tests → exit 0, `14 passed in 1.40s`;
full `./test.sh` → exit 0, `113 passed in 1.50s`.
Next: agree on throughput timing boundaries and add a deterministic-clock test.
Throughput, scheduled evaluation, GPU behavior, and remote delivery remain
unverified. Earlier records below are historical.

Previous Red: `count_tokens=True` with optional
`pad_token_id`, replacing the callback proposal. Builder/Trainer default to
disabled counting and no padding id. Added the first unpadded LM integration
test: 8/8/4 input tokens must emit integer totals 16/20 at steps 2/3, with
real cross-entropy/SGD updates. Baseline: `107 passed in 2.12s`.
Targeted logging tests: `1 failed, 8 passed in 0.99s`; full `./test.sh`:
`1 failed, 107 passed in 1.32s`. Intended Red: Builder rejects `count_tokens`;
new training/payload assertions have not executed. Only tests/docs changed.
Historical next action: add/wire the optional settings and log token totals in both
event paths, then verify Green. Padding, lifetime cumulative counting,
evaluation exclusion, disabled-count compatibility, and throughput follow.

Latest verification: parameterized the one-update global L2 case to include an
unused trainable parameter and a frozen parameter. Each stays at `100`, retains
`grad is None`, and does not change the expected norm `sqrt(104)` or real SGD
updates. `./test.sh tests/integration/test_training_logging.py` → exit 0,
`8 passed in 1.46s`; full `./test.sh` → exit 0, `107 passed in 1.39s`.
Both new cases passed on first execution; no Red is claimed. Only tests/docs
changed. Next: agree on explicit LM token counting and interval timing before
testing `train/total_tokens_seen` and `train/tokens_per_second`. No new API is
agreed; generic regression inputs must not automatically count as tokens.

Latest verification: extended the existing 2/3/4-update integration cases to
assert last-update gradient norms at interval boundaries and the final partial
interval, plus unchanged final gradients and correct SGD weights. Expected
norms at steps 2/3/4 are `3.2`, `2.56`, and `2.048`.
`./test.sh tests/integration/test_training_logging.py` → exit 0,
`6 passed in 1.55s`; full `./test.sh` → exit 0, `105 passed in 1.37s`.
All added assertions passed on first execution; no Red is claimed. Only
tests/docs changed. Next: cover parameters without gradients, then agree on
token-count/throughput policy before adding new metric tests.

Latest implementation: Trainer computes global
gradient L2 after backward and before the optimizer update, leaving gradients
unchanged. Both interval and final-partial events include `train/grad_norm`
from the last update. Scalar conversion occurs only when emitting an event.
`./test.sh tests/integration/test_training_logging.py` → exit 0,
`6 passed in 1.35s`. Full suite not rerun. Next: add multi-step/final-partial
norm assertions; clipping/AMP and remote delivery remain unverified.
Earlier records below are historical.

Latest increment: added the one-update global L2 `train/grad_norm` test using
real SGD, two weights, and a bias. Expected `sqrt(104)` from gradients
`[-6, -8, -2]`; update/gradient/loss/LR assertions pass, then the metric lookup
raises `KeyError: 'train/grad_norm'`. Baseline: `104 passed in 1.80s`.
Targeted logging tests: `1 failed, 5 passed in 1.14s`; full `./test.sh`:
`1 failed, 104 passed in 1.39s`. Only tests/docs changed.
Next: user logs the global L2 norm without mutating gradients, then verify Green.
Multi-step norm sampling, actual clipping timing, and final-partial norm logging
are not established by this one-update case. Earlier records are historical.

Latest: at the user's explicit implementation request, Trainer now captures
single-group LR before optimizer updates and logs `train/learning_rate` beside
loss in both interval and final-partial events. Targeted logging test:
`5 passed in 1.66s`; full `./test.sh`: `104 passed in 1.66s`.
Loss and single-group constant-LR logging are Green. Next: agree on gradient
norm sampling (proposed global L2 before clipping on the last completed update),
then add its smallest test. Multi-group LR naming, token/throughput policies,
automatic evaluation inputs/schedule, and remote delivery remain pending.
This implementation increment changed Trainer/docs; earlier results follow.

The user requested coverage of the screenshot's seven metrics. The next small
increment is single-group `train/learning_rate` in the existing loss events,
including the final partial interval. Baseline: `102 passed in 1.88s`.
New real-SGD cases (LR 0.1/0.05) confirm updates and steps 2/3, then fail with
`KeyError: 'train/learning_rate'`. Targeted logging test: `2 failed, 3 passed
in 1.46s`; full `./test.sh`: `2 failed, 102 passed in 1.68s`.
Only tests/docs changed. Next: user adds LR to both event paths, then verify
Green. Gradient norm, multi-group keys, token/throughput policy, and automatic
eval inputs/schedule still require agreement before tests. Loss logging is
Green; this does not establish logging of the other metrics or remote delivery.
Earlier status/next-action entries below are historical.

Current files default both Config and Trainer intervals to `None`; baseline
`./test.sh` is Green: `100 passed in 1.91s`. Added exact-interval cases (2 and
4 updates at interval 2) alongside the existing partial-interval case. All pass
on first execution: logging/Builder/config regression command →
`13 passed in 1.48s`. No artificial Red; the full suite was not rerun after
these test-only changes. Only tests/docs changed in this increment.
Next: verify training with an explicit interval and no logger, then agree on
interval validation. Earlier failures and next-action entries below are historical.

Latest full suite after the user set the config default to `10`:
`./test.sh` → `2 failed, 98 passed in 1.99s`. Both built-Trainer cases expect
disabled logging when omitted, but the default enables the final partial flush.
Next: default the interval to `None` in config and Trainer, then rerun the suite.
This diagnostic follow-up changed only documentation; earlier results follow.

Follow-up full suite: `./test.sh` → `5 failed, 95 passed in 1.86s`.
All five fail because existing callers omit the newly required
`TrainingConfig.logging_steps`; interval logging passes. Next: user defaults
the config interval to `None`, aligns Trainer's default for direct callers, and
reruns the suite. This follow-up changed only documentation.

Latest check: at the user's explicit implementation request, Trainer now emits
interval mean loss and flushes the final partial interval. Logging plus Trainer
loss/empty/evaluation tests pass: `12 passed in 1.54s`; full suite not run.
Config and Builder wiring now reach the loop. Existing defaults still need to
restore agreed omitted-setting compatibility (TrainingConfig requires the
interval; Trainer defaults to 10). Next: resolve those defaults and cover exact
interval completion/disabled logging. The initial Red recorded below is historical.

On branch `feat/Trainer-Log`, HEAD `6476e04` (minimum evaluation committed), the
user selected `logging_steps` on `TrainingConfig`, wired through Builder. First
behavior: emit the mean of step losses every N completed optimizer updates and
flush the final partial interval; omitted setting preserves disabled automatic
logging. First integration test is Red: `TrainingConfig` rejects `logging_steps`.
Latest `./test.sh` on 2026-11-13 → `1 failed, 99 passed in 1.83s`; baseline before
the test was `99 passed in 2.07s`. Next: user adds/configures the interval and
implements loop emission, then verify Green. Further metrics, cumulative remote
steps, token-count policy, and scheduled eval are planned in 019, not yet tested.
The agent changed only tests/docs. The self-written eval follow-up below is
deferred while working on train-loop logging.

Previous direction (deferred): [017 — Self-Written Evaluation](017-small-evaluation.md).
On 2026-11-13 the user stopped [018's Flock evaluation integration](018-flock-final-evaluation.md)
and requested a self-written evaluator. Reuse `Trainer.evaluate`; the scope
question is pending: FineWeb-Edu dev loss/PPL first, or all three benchmark
families. No new evaluation API/tests or dependencies are introduced until the
scope is clear. Flock logging in task 013 is separate. Do not delete the user's
Flock checkout or continue its setup. Most recent full suite: `99 passed in
1.96s` (before this scope change); current targeted result is recorded in 017.
Next: settle custom evaluation scope and data preprocessing, then identify the
smallest missing behavior for the next TDD step. The supplied lab requirements
still specify Flock for official scoring; acceptance of a substitute is unconfirmed.

Previous task: [017 — Small Development Evaluation](017-small-evaluation.md).
The user accepted a small local dev evaluation and final scoring through Flock.
Agreed first API: `Trainer.evaluate(batches)` returns `{"loss": float,
"num_targets": int}` using target-weighted mean loss for dense, unpadded targets
and a mean-reduced supplied loss; the LM caller derives perplexity.
On branch `feat/Trainer-Log` at `1d2dc74` (PR #6 merged), the user implemented
`evaluate` and moved mode restoration to `finally`. Latest checks on 2026-11-13:
targeted eval/training/LM command → `11 passed in 1.67s`; full `./test.sh` →
`99 passed in 1.38s`. Weighted loss/types, eval mode/no-grad, normal/loss-error
mode restoration, empty list/iterator rejection, and preservation of parameters,
existing gradients, populated optimizer state, and training progress are Green.
The tiny LM now uses public `evaluate` before/after real training: 64 targets,
initial loss above 1, final loss below 0.05. New cases passed on first run;
no artificial Red is claimed. The agent changed only tests/docs.
Next: prepare a fixed real dev subset and agree on its preprocessing, then verify
manual evaluation with a real causal LM before agreeing on automatic scheduling.
Flock execution, GPU/masking/distributed behavior, and real dev metrics remain
unverified. Detailed commands and historical failures are in 017.
The older branch/next-action records below are historical context.

Flock eval documentation was successfully read on 2026-11-12 at documentation
revision `710e75a857969be3fefef67c50b3df016f15ef15`; details and links are in 017.
Important: official `fineweb_edu` scores documents separately with 8192-token
windows and rejects a shorter model context. A local 4096 packed eval remains a
development diagnostic. Limited in-memory scoring is available through
`prepare_task(..., limit=N)` and `score_prepared`. No Flock execution, package
installation, or model export compatibility has been verified. Confirm the
planned model can support 8192 before a long training run.

Read [Architecture.md](../docs/Architecture.md), [Trainer.md](../docs/Trainer.md), and [013 — Flock Logger and Delivery Verification](013-flock-logger.md), then inspect the relevant source and tests. Former tasks 013 and 014 are merged into task 013. The agreed public class is `upwash.core.logger.FlockLogger`, implemented by the user in `core/logger/flock.py` and re-exported by `core/logger/__init__.py`. The test is `tests/unit/core/test_flock_logger.py`. The user writes production behavior; the agent supports design, tests, failure analysis, and documentation under [AGENTS.md](../AGENTS.md).

## Current Position

As of 2026-11-12, PRs #1–#5 are merged. `main` is at `1fca7a3` (PR #5, FlockLogger).

Configuration (010), MemoryLogger (011), Builder (012), and the three FlockLogger unit cases are now verified Green. The user added deep-copying before SDK forwarding; all 65 tests pass. Task 013 remains incomplete because actual Flock delivery has not been exercised: current tests substitute RecordingRun for the SDK run.

The accepted constructor is `FlockLogger(project, name, config=None)` and initializes the SDK run from those settings; `log(metrics, *, step)` forwards events. The caller supplies experiment settings and owns logging policy, Flock host configuration, and authentication. The user implemented finish() and an atexit fallback; repeated-finish and post-finish behavior remain untested. Builder passes through the optional logger and Trainer stores it without automatic metric emission. The official Flock guide confirms use of the wandb SDK; W&B service support is outside scope. No SDK serialization or Flock delivery has been verified. Only finishing the unit test does not complete task 013.

```python
logger = FlockLogger(project="lab6-predictable-scaling", name="tiny-training", config={"lr": 0.1})
```

The earlier `FlockLogger(run=run)` constructor is superseded. Unit tests replace `wandb.init` with a RecordingRun factory; they do not initialize a real SDK run.

The required Flock scenario logs these seven metrics: `train/loss`, `train/grad_norm`, `train/learning_rate`, `train/tokens_per_second`, `train/total_tokens_seen`, `eval/loss`, and `eval/perplexity`. Their definitions and known-value payload are recorded in [013](013-flock-logger.md). Unit tests verify forwarding of supplied values; their numerical calculation remains with training/evaluation code. No automatic Trainer metric emission or logger key filtering is added. Parameter-group LR naming remains to be agreed for a future multi-group case.

Data area, current task: [015 — Read many shards as one dataset, with a sample budget](015-multi-shard-dataset.md), on branch `feat/multi-shard-dataset`. Agreed interface: `ConcatDataset(datasets)` and `Truncate(dataset, num_samples)`. Red verified on 2026-11-12: `ImportError` for both names. The user implemented both classes, and Green was reached the same day: `21 passed` for the task, full suite `83 passed`. A read-only check on the real shard gives the expected lengths and `(8, 4096)` batches. Committed as `41f8bca`.

Next data task, in the same pull request: [016 — Read Flock's pre-tokenized `.npy` shards](016-npy-token-shard.md). Agreed interface: `PackedTokenDataset.from_npy(path, seq_len)`. Red verified on 2026-11-12: six new cases failed with `AttributeError` (no `from_npy`). The user implemented it, and Green was reached the same day: `21 passed` in the file, full suite `89 passed`. A read-only check on a real Flock `tokens.npy` confirms memory-mapping, `len` 15,395, and `(8, 4096)` batches. Still open: verifying the Flock data's dev-set exclusion. Committed as `65f0833`. Both commits are pushed on `feat/multi-shard-dataset`, and `main` (with PR #5) was merged into the branch on 2026-11-12; the only conflict was this README. Next: open the pull request for 015 and 016, then seeded shuffle.

## Current Verification — 2026-11-12

Latest after the user's mapping-copy implementation: `./test.sh` → exit 0; `65 passed in 1.95s`. Constructor, seven-metric forwarding (values/step/commit/input isolation), and exit-callback regression are Green. This supersedes the earlier Red records below. All Flock tests patch wandb.init and use RecordingRun; no real SDK run or upload occurs, even with project lab6-predictable-scaling. Source search found no separate real Flock logging scenario under src/scripts/tests. The requested dashboard URL could not be inspected with the web tool, so no claim is made about existing server runs. Next: prepare actual Flock delivery verification with the selected organization/project, host/key setup, a distinct run name, and explicit completion; obtain authorization for the concrete hosted upload before execution. No credentials were read or changed and no network logging occurred.

Latest after explicit commit forwarding: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.23s`. The metric-event test now verifies all seven metric values, step 7, commit=True, and unchanged caller input immediately after log. Its final assertion fails: clearing the caller's metrics also clears RecordingRun's received mapping (`{}` instead of seven values). This confirms the intended mapping-isolation Red. Next: user forwards a separate dict to the SDK, preserving the current step/commit behavior. The full suite was not rerun. Earlier commit/export failures below are historical. The user also added an optional commit argument; commit=False behavior has not been agreed or tested.

Latest after the user's package export: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.25s`. Constructor and exit callback pass. Metric forwarding now reaches `flock.py:20` and raises `TypeError: RecordingRun.log() missing 1 required keyword-only argument: 'commit'` because production calls `run.log(metrics, step=step)`. This is the intended forwarding Red for the agreed explicit commit behavior. Input-isolation assertions have not executed; the source still passes the caller's mapping directly. Next: user forwards an independent mapping and `commit=True`, then rerun the targeted tests. The full suite was not rerun for this check. Earlier missing-export outcomes below are historical.

Latest cleanup verification: `./test.sh tests/unit/core/test_flock_logger.py` → `1 failed, 2 passed in 0.24s`; `./test.sh` → `1 failed, 64 passed in 1.80s` (both exit 1). The sole failure is the missing FlockLogger package export. Constructor and captured exit-callback cases pass; no ignored atexit exception appeared. The error was a missing finish() method on the RecordingRun test double, reproduced Red and then fixed in test support. Tests intercept atexit registration to prevent process-exit callbacks leaking across cases. Production code was unchanged; see [013](013-flock-logger.md) for commands and the regression cause.

The earlier table below predates the cleanup regression:

| Command | Result | Interpretation |
| --- | --- | --- |
| `./test.sh tests/unit/core/test_flock_logger.py tests/unit/core/test_logger.py` | Exit 1; `1 failed, 5 passed in 0.28s`. | Required-project constructor and four MemoryLogger cases pass. Metric test is Red for the missing package export; seven-key payload assertions have not executed. |
| `./test.sh tests/unit/core/test_flock_logger.py` (earlier snapshot) | Exit 1; `2 failed in 0.02s`. | Historical syntax blocker, now resolved. |
| `./test.sh` (earlier project/payload) | Exit 1; `1 failed, 63 passed in 2.04s`. | Earlier regression evidence only; full suite not rerun after updating the fixture. |

Tests now use project `lab6-predictable-scaling` and the exact seven metric names from the user's requirements. Both logger cases patch SDK initialization and create no real run. The agent changed only tests and documentation; dependencies, source, credentials, and SDK settings were not changed.

Next: user exports FlockLogger through `core/logger/__init__.py` and completes metric forwarding, then rerun the targeted tests. Use explicit finish() when verifying delivery with the same project and metrics; its broader lifecycle semantics still need tests. Default/empty-config behavior, metric computation, real SDK serialization, and actual Flock delivery remain unverified. SDK is 0.31.0; the server/environment and exact deployed version remain to be selected.

## Route 011–013

| Task | Deliverable | Dependencies | Next action |
| --- | --- | --- | --- |
| [011](011-metric-logger.md) | Shared logging interaction and MemoryLogger. | 010 complete. | Complete for its four tested memory behaviors. |
| [012](012-build-trainer.md) | Build configured optimizer and Trainer with optional logger storage. | 010 complete. | Complete; existing Builder tests remain Green. |
| [013](013-flock-logger.md) | FlockLogger(project, name, config=None) and actual Flock delivery. | 011; scheduled after 012. | Constructor and exit callback Green; All unit cases Green; next verify actual Flock delivery. |

Former task 014 is merged into 013 and is no longer a separate milestone. Earlier tasks keep their numbers.

## Shared Design

- FlockLogger initializes its SDK run from `project`, `name`, and optional `config`, then exposes `log(metrics, *, step)` and forwards an independent metric mapping with the supplied step and `commit=True`. Only MemoryLogger exposes `records`.
- The caller supplies experiment settings and detached scalar metrics, and chooses logging frequency and valid increasing SDK steps. FlockLogger initializes the run; the user has implemented finish() with an atexit fallback; only the single callback path is tested. Returned `state.global_step` remains per-call; no cumulative Trainer logging counter is planned.
- Builder passes an optional logger unchanged to Trainer. Ordinary core/MemoryLogger imports must not require loading the remote SDK.
- A fresh Flock run created through the logger is the first delivery scenario. Resume, shared runs, CompositeLogger, multi-host delivery, and retry policies remain deferred.

Official guide links, SDK/documentation versions, historical adapter results, and delivery acceptance steps live in [013](013-flock-logger.md). Only an actual server check establishes delivery; offline SDK serialization does not.

## Historical Verification — Earlier on 2026-11-12

These entries predate the FlockLogger rename and merged task. Their old names, paths, and next-action notes are historical; the current verification above supersedes them.


- Latest full suite: `./test.sh` → exit 1; `1 failed, 62 passed in 2.28s`. Sole intended Red: `ImportError` for the missing `WandbLogger` in `upwash.core.logger`; metric assertions have not executed. All existing tests, including the data suite, pass. Earlier environment-blocker results below are historical.
- Local packages now include wandb 0.31.0, huggingface_hub 2.4.0, pyarrow 26.0.1, and pytest 9.2.0. The user's `pyproject.toml` declares `wandb>=0.31.0`; the agent made no dependency, lockfile, installation, or test-configuration changes. The user has since renamed the source to `flock.py`; it imports wandb but contains no logger class.
- Documentation read: the user-supplied hosted guide and its logging/compatibility references were successfully retrieved (HTTP 200), resolving the previous document-access blocker. The documented SDK transport is confirmed; deployed compatibility remains unverified. The pages identify documentation revision `710e75a857969be3fefef67c50b3df016f15ef15`. No tests were rerun and no source/test changes were made for this documentation-only update.
- Current branch is `main`, with PR #3 merged at `7e60792`. The worktree was clean before this session's documentation updates.
- `./test.sh tests/unit/test_training_config.py tests/unit/core/test_logger.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → exit 0; `14 passed in 2.95s`. Tasks 010–012 remain Green for their tested scope.
- `./test.sh` → exit 2; eight collection errors because the local `.venv` lacks `huggingface_hub` and `pyarrow`. This is an environment blocker, not an intended Red or a verified full-suite regression. No dependencies or test configuration were changed.
- Historical first Red at the superseded `integrations/` path: `./test.sh tests/unit/integrations/test_wandb_logger.py` → exit 1; `1 failed in 0.01s` for missing `upwash.integrations`. The user subsequently agreed to place WandbLogger in `core/logger.py`, and the test moved to `tests/unit/core/test_wandb_logger.py`. It expects one complete metric event with step 7, `commit=True`, and input isolation.
- Rechecked the same 010–012 command above after adding the test → exit 0; `14 passed in 1.29s`.
- After agreeing on `core/logger.py`: `./test.sh tests/unit/core/test_wandb_logger.py tests/unit/core/test_logger.py tests/unit/test_training_config.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → exit 1; `1 failed, 14 passed in 1.45s`. Intended Red: cannot import the missing `WandbLogger` class from the existing `upwash.core.logger` module. Behavioral assertions have not executed; existing regression cases remain Green.
- After the user-requested conversion to `core/logger/`, the same command → exit 1; `1 failed, 14 passed in 1.31s`. MemoryLogger's public import and existing behavior remain Green, and its moved source is byte-for-byte unchanged. WandbLogger remains Red for its missing class/export in the package. No SDK or dependency changes were made.
- Next: user implements and exports the run adapter in `flock.py`, then verify Green and select the Flock server/environment for 014. The last agreed public class is WandbLogger; align any rename before editing tests. An offline SDK check is an optional diagnostic, not a separate W&B milestone. Official documentation confirms the selected transport; actual delivery still requires execution against Flock.

## Historical Verification — 2026-11-09


| Command | Result | Interpretation |
| --- | --- | --- |
| `./test.sh tests/unit/test_training_config.py` | Exit 0; `6 passed in 0.01s`. | Current Config tests Green: composition, positive/zero values, negative rejection, NaN and both infinities. Broader input-type behavior is deferred. |
| `./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py` | Exit 0; `4 passed in 2.05s`. | Construction, logger omission, real SGD updates at lr 0.1/0.05, and no automatic logging verified. |
| `./test.sh` | Exit 0; `46 passed in 2.11s`. | Current combined suite Green. |

Earlier environment and optional-logger failures are resolved. The two new integration cases passed on their first run; no Red is claimed for already-supported behavior. Detailed historical results remain in task files. The agent changed no dependencies or test configuration.

## Data Pipeline Handoff

Tasks [007](007-load-dataset.md) and [008](008-write-token-shard.md) retain their existing scope and records. Task 008 includes dev-id loading, resumable per-file shard generation, and `scripts/prepare_fineweb.py`. A recorded real-file run produced 726,000 documents and 731,876,929 tokens with matching shard-size and EOS-count checks. Its review updates record one snapshot request, shared parquet reading, and explicit little-endian `"<u4"` tokens, with historical Docker full-suite result `32 passed`.

Task 008 merged into `main` through PR #2. The skip-on-rerun check of the script has not been reported to the agent yet.

Task [009](009-packed-token-dataset.md) merged into `main` through PR #3. `PackedTokenDataset(path, seq_len)` memory-maps a `"<u4"` shard and returns `int64` `(x, y)` next-token pairs for `DataLoader`. A tiny LM trained through `DataLoader` and `Trainer` reaches loss below `0.05`. The real 008 shard with `seq_len=4096` yields 178,680 samples and `(8, 4096)` batches. On 2026-11-09, `main` (with tasks 010–012) was merged into the branch. The only conflict was this README. Full suite (Docker): `54 passed`. PR #3 review then asked for argument and index checks. `seq_len <= 0` now raises `ValueError`, negative indices count from the last sample, and out-of-range indices raise `IndexError`. Green: `15 passed` in the 009 unit file, full suite `62 passed`. PR #3 merged without the two suggested refinements. Both are still open and optional: check `seq_len` before opening the shard, and report the caller's index in the `IndexError` message. Only one real shard exists locally (`sample/10BT/000_00000.bin`, about 0.73B tokens).

## Gaps Toward the Lab Run

The lab target is a 1B Llama-3.2-style model trained on 200B FineWeb-Edu tokens with context 4096 and evaluated by Flock. This list is the agent's gap analysis against `main`. It has not been agreed, ordered, or assigned.

- Data: a seeded shuffle (reading many shards and a token budget are done in task 015), a resume position, per-rank sharding, and tokenizing about 200B tokens. One file took about 31 minutes for 0.73B tokens, so a single process needs on the order of 140 hours. Task 016 instead reads Flock's pre-tokenized `.npy` shards, synced to local disk.
- Model: no Llama implementation yet (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings), and no decision between writing one and depending on one.
- Training loop: no per-step loss output, learning-rate schedule (warmup and decay), gradient clipping, gradient accumulation, weight-decay parameter groups, bf16 mixed precision, device placement, checkpoint save and resume, multi-GPU training, or validation loss.
- Logging: FlockLogger unit cases are Green (013, PR #5); actual delivery to Flock is unverified. Useful metrics depend on `train()` exposing a per-step loss.
- Evaluation hand-off: the checkpoint format Flock evaluation expects has not been confirmed.
- Optional: μP (per-group learning rates, output multiplier, coordinate check). It covers width only, and the lab has not said whether it is required.

Tasks 001–008 keep their numbers, and 009 is the packed-token dataset. Configuration and logging originally used 010–014; former task 014 is now merged into 013.

## Completed Route

| Task | Behavior | Evidence | Status |
| --- | --- | --- | --- |
| [001](001-kernel-loss.md) | Compute scalar loss using supplied MSE and L1 loss functions. | Expected values `4` and `2`; tests now supply the required optimizer. | Green. Original API description is historical. |
| [002](002-minimal-training-loop.md) | Run a minimal training loop. | Real updates, fresh gradients, training mode, and per-call progress across successive calls. | Green, including repeated-call progress; supersedes former tasks 002–005. |
| [003](003-model-forward-hooks.md) | Use model output modified by a forward hook. | Hook-modified loss `1`; full suite `6 passed`. | Green. |
| [004](004-empty-batches.md) | Reject iterables that yield no batches. | Empty list and iterator raise `ValueError`; full suite `8 passed`. | Green. |
| [005](005-data-loader.md) | Minimal dataset and loader consumed by `Trainer.train`. | Ordered batches `[2, 2, 1]`, `drop_last`, `len` including divisible cases, repeated iteration, trained weight `0.72`. | Green: full suite `18 passed`. |
| [006](006-fit-linear-regression.md) | Fit `y = 3x + 1` end to end with the existing framework. | Final loss below `1e-4`; weight and bias within `1e-3` of `3` and `1`. | Verified: `19 passed`; no production change needed. |
| [007](007-load-dataset.md) | Read selected columns from pinned parquet files on the Hugging Face Hub, reusing the local cache. | Matching files in sorted order, requested columns, `ValueError` when nothing matches, pinned snapshot request. | Green: `23 passed`. |
| [008](008-write-token-shard.md) | Exclude dev ids, tokenize, append EOS, write a `uint32` token shard. | Exact token sequence, counts, no partial file on failure. | Green (`26 passed`); manual real-data check: 1,000 documents, 1,037,837 tokens, exact decode round trip. Wiring: `31 passed`; later snapshot-request and explicit little-endian review changes: `32 passed` in Docker. |
| [009](009-packed-token-dataset.md) | Read a `"<u4"` shard as packed `(x, y)` next-token samples for `DataLoader`. | Exact samples and shared boundary token, `len` cases, `int64` with large ids, tiny LM loss below `0.05`; `ValueError` for `seq_len <= 0`, negative indices, `IndexError` out of range, iteration stops at `len`. | Green: `40 passed` on the branch; `54 passed` after merging `main`; `62 passed` with review checks. Real shard yields 178,680 samples of 4,096 tokens. |
| [010](010-training-config.md) | Composed optimizer settings with finite nonnegative learning rate. | Six config cases pass locally on 2026-11-09; prior branch-local full suite `25 passed`. | Green for tested Python-float scope. |
| [011](011-metric-logger.md) | Batch scalar metrics in memory, ordered independent records, input isolation. | Four logger tests; full suite `42 passed` on 2026-11-09. | Green for tested memory scope. |
| [012](012-build-trainer.md) | Build configured optimizer and Trainer with optional logger storage. | Two unit cases and two real-SGD integration cases; full suite `46 passed`. | Green; no automatic metric emission. |
| [015](015-multi-shard-dataset.md) | Read many shards as one dataset and keep the first `num_samples` samples. | Ordered index mapping, empty parts skipped, negative and out-of-range indices, no crossing between real shards, budget checks. | Green: `21 passed`, full suite `83 passed`. |
| [016](016-npy-token-shard.md) | Read Flock's `int32` `.npy` shards with `PackedTokenDataset.from_npy`. | Same samples as `.bin`, `int64` with large ids, `seq_len` and index checks, two `.npy` shards in `ConcatDataset`; real Flock file memory-mapped. | Green: `21 passed` in the 009 file, full suite `89 passed`. |

Detailed implementation iterations, test commands, and failures remain in each task document. Returned-state snapshot semantics and partial-failure recovery remain deferred. Custom operation kernels are not required for this milestone.

## Maintaining This Record

Keep the current task, observed results, blockers, and next action here. Keep detailed evidence in task files. Label API proposals, implemented behavior, historical results, and current verification separately. Do not count an unrun test as Green or an environment failure as the intended Red. Update the roadmap after each meaningful TDD step.
