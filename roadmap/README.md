# Implementation Progress

Last updated: 2026-11-12 (Asia/Taipei).

## Resume Here

Read [Architecture.md](../docs/Architecture.md), [Trainer.md](../docs/Trainer.md), and [013 — Shared wandb SDK adapter](013-wandb-logger.md), then inspect the relevant source and tests. The initial MemoryLogger batch interface is agreed; the task 012 Builder interface with optional logger injection is also agreed; remote adapter proposals still need agreement before tests. The user writes production code; the agent supports design, tests, failure analysis, and documentation under [AGENTS.md](../AGENTS.md).

## Current Position

As of 2026-11-09, PRs #1–#4 are merged. `main` is at `7e60792`, and the full suite on it returns `62 passed` (Docker). Completed: tasks 001–012. Not started: 013 and 014 (remote logging) and the training gaps listed under [Gaps Toward the Lab Run](#gaps-toward-the-lab-run).

Data area, current task: [015 — Read many shards as one dataset, with a sample budget](015-multi-shard-dataset.md), on branch `feat/multi-shard-dataset`. Agreed interface: `ConcatDataset(datasets)` and `Truncate(dataset, num_samples)`. Red verified on 2026-11-12: `ImportError` for both names. The user implemented both classes, and Green was reached the same day: `21 passed` for the task, full suite `83 passed`. A read-only check on the real shard gives the expected lengths and `(8, 4096)` batches. Next: commit and open a pull request, then seeded shuffle.

Configuration (010), MemoryLogger (011), and Builder (012) are implemented and verified Green. Current task is [013 — Shared wandb SDK adapter](013-wandb-logger.md): agree on adapter API, exact SDK version, and dependency/test setup before the first test. The same adapter is planned for W&B or Flock; SDK compatibility and actual server delivery remain unverified.

Builder constructs the configured optimizer for the supplied model and passes through an optional logger. Trainer stores `trainer.logger`; metric selection, timing, and actual logging remain with the user's training logic. Two real SGD integration cases verify different learning rates update the supplied model and training emits no automatic metrics. `core/contracts.py` remains empty; introduce shared typing when a tested interaction needs it. Tasks 013–014 have no tests or implementation. The agent changed only tests and documentation during this step.

## Current Verification — 2026-11-09

| Command | Result | Interpretation |
| --- | --- | --- |
| `./test.sh tests/unit/test_training_config.py` | Exit 0; `6 passed in 0.01s`. | Current Config tests Green: composition, positive/zero values, negative rejection, NaN and both infinities. Broader input-type behavior is deferred. |
| `./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py` | Exit 0; `4 passed in 2.05s`. | Construction, logger omission, real SGD updates at lr 0.1/0.05, and no automatic logging verified. |
| `./test.sh` | Exit 0; `46 passed in 2.11s`. | Current combined suite Green. |

Earlier environment and optional-logger failures are resolved. The two new integration cases passed on their first run; no Red is claimed for already-supported behavior. Detailed historical results remain in task files. The agent changed no dependencies or test configuration.

## Route 011–014

| Task | Deliverable | Dependencies | Next action |
| --- | --- | --- | --- |
| [011](011-metric-logger.md) | Shared logging interaction and MemoryLogger. | 010 complete. | Complete for four tested memory behaviors; full suite `42 passed`. |
| [012](012-build-trainer.md) | Build an optimizer from Config and wire Trainer. | 010 complete. | Complete; four Builder cases pass, full suite `46 passed`. |
| [013](013-wandb-logger.md) | One wandb SDK adapter for W&B or Flock. | 011; scheduled after 012. | Current: agree adapter API, exact SDK version, and optional dependency setup. |
| [014](014-flock-logger.md) | Verify actual delivery to Flock through that adapter. | 013. | Select pinned SDK/server versions and an agreed integration environment. |

Each task uses small Red/Green steps. Builder constructs the optimizer and Trainer and passes through an optional logger; metric emission stays with the caller. No Hugging Face class replacement or differential compatibility scope is selected for this route.

## Shared Design Proposals

The initial batch logging interface and memory record representation are agreed. The caller owns logging policy; remote adapter details remain proposals:

- All backends support `log(metrics, *, step)`; only MemoryLogger exposes `records`.
- The calling training logic supplies detached scalar metrics, chooses logging frequency, and manages steps suitable for the destination. Returned `state.global_step` remains per-call; no automatic cumulative Trainer logging counter is planned.
- Caller owns remote run initialization, host/authentication settings, and completion. Builder accepts optional `logger=None` and injects it unchanged into Trainer, which stores `trainer.logger`; neither initializes or finalizes remote runs, and core imports no concrete remote SDK.
- One caller-owned fresh run is the first remote scenario. Resume, shared runs, CompositeLogger, simultaneous multi-host delivery, and failure/retry policies require later requirements and agreement.

Offline SDK tests establish serialization; only a real server check establishes delivery. Flock evidence and source links live in [014](014-flock-logger.md), SDK adapter evidence in [013](013-wandb-logger.md). Exact versions must be recorded before executable compatibility checks.

## Data Pipeline Handoff

Tasks [007](007-load-dataset.md) and [008](008-write-token-shard.md) retain their existing scope and records. Task 008 includes dev-id loading, resumable per-file shard generation, and `scripts/prepare_fineweb.py`. A recorded real-file run produced 726,000 documents and 731,876,929 tokens with matching shard-size and EOS-count checks. Its review updates record one snapshot request, shared parquet reading, and explicit little-endian `"<u4"` tokens, with historical Docker full-suite result `32 passed`.

Task 008 merged into `main` through PR #2. The skip-on-rerun check of the script has not been reported to the agent yet.

Task [009](009-packed-token-dataset.md) merged into `main` through PR #3. `PackedTokenDataset(path, seq_len)` memory-maps a `"<u4"` shard and returns `int64` `(x, y)` next-token pairs for `DataLoader`. A tiny LM trained through `DataLoader` and `Trainer` reaches loss below `0.05`. The real 008 shard with `seq_len=4096` yields 178,680 samples and `(8, 4096)` batches. On 2026-11-09, `main` (with tasks 010–012) was merged into the branch. The only conflict was this README. Full suite (Docker): `54 passed`. PR #3 review then asked for argument and index checks. `seq_len <= 0` now raises `ValueError`, negative indices count from the last sample, and out-of-range indices raise `IndexError`. Green: `15 passed` in the 009 unit file, full suite `62 passed`. PR #3 merged without the two suggested refinements. Both are still open and optional: check `seq_len` before opening the shard, and report the caller's index in the `IndexError` message. Only one real shard exists locally (`sample/10BT/000_00000.bin`, about 0.73B tokens).

## Gaps Toward the Lab Run

The lab target is a 1B Llama-3.2-style model trained on 200B FineWeb-Edu tokens with context 4096 and evaluated by Flock. This list is the agent's gap analysis against `main`. It has not been agreed, ordered, or assigned.

- Data: a seeded shuffle (reading many shards and a token budget are done in task 015), a resume position, per-rank sharding, and tokenizing about 200B tokens. One file took about 31 minutes for 0.73B tokens, so a single process needs on the order of 140 hours.
- Model: no Llama implementation yet (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings), and no decision between writing one and depending on one.
- Training loop: no per-step loss output, learning-rate schedule (warmup and decay), gradient clipping, gradient accumulation, weight-decay parameter groups, bf16 mixed precision, device placement, checkpoint save and resume, multi-GPU training, or validation loss.
- Logging: the wandb adapter (013) and Flock delivery (014). Both depend on `train()` exposing a per-step loss.
- Evaluation hand-off: the checkpoint format Flock evaluation expects has not been confirmed.
- Optional: μP (per-group learning rates, output multiplier, coordinate check). It covers width only, and the lab has not said whether it is required.

Tasks 001–008 keep their numbers. The configuration and logging tasks moved to 010–014, and 009 is the packed-token dataset.

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

Detailed implementation iterations, test commands, and failures remain in each task document. Returned-state snapshot semantics and partial-failure recovery remain deferred. Custom operation kernels are not required for this milestone.

## Maintaining This Record

Keep the current task, observed results, blockers, and next action here. Keep detailed evidence in task files. Label API proposals, implemented behavior, historical results, and current verification separately. Do not count an unrun test as Green or an environment failure as the intended Red. Update the roadmap after each meaningful TDD step.
