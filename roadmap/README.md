# Implementation Progress

Last updated: 2026-11-08 (Asia/Taipei).

## Resume Here

Read [the architecture](../docs/Architecture.md), [the Trainer goal](../docs/Trainer.md), and [the latest task](008-write-token-shard.md), then inspect relevant source and tests. Verify this record against the checkout before proceeding.

## Current Position

Current task: [009 — Read a token shard as packed training samples](009-packed-token-dataset.md), on branch `feat/packed-token-dataset` (created from `feat/load-dataset` while that pull request awaits review). Agreed interface: `PackedTokenDataset(path, seq_len)`, which returns `int64` `(x, y)` next-token pairs from a memory-mapped `uint32` shard. The user implemented it; Green on 2026-11-08 (`8 passed`; full suite `39 passed`). A tiny LM trained through `DataLoader` and `Trainer` reaches loss below `0.05`. On the real 008 shard with `seq_len=4096`, there are 178,680 samples and batches of shape `(8, 4096)`. Next: commit and open a pull request; then the deferred data features (multiple shards, seeded shuffle, token budget, resume, distributed sharding).

Previous task, 008 below: the pull request is pushed and awaiting review. The result of the skip-on-rerun check has not been reported to the agent yet.

Current task: [008 — Tokenize documents into a token shard](008-write-token-shard.md), continuing on branch `feat/load-dataset`. `write_token_shard` excludes dev-set ids, appends EOS per document, and writes `uint32` tokens atomically. The user implemented it; Green on 2026-11-08 with `26 passed`. A manual run with the real Llama 3 tokenizer, `dev-ids.txt`, and 1,000 FineWeb-Edu documents produced 1,037,837 tokens and an exact decode round trip. Wiring was folded into this task: `load_dev_ids` and `prepare_token_shards` (one `.bin` plus sidecar `.json` per parquet file, skipping completed shards). The user implemented both; Green on 2026-11-08 (`5 passed`; full suite `31 passed`). The user wrote `scripts/prepare_fineweb.py` (all revisions pinned, including `dev-ids.txt` at `2484afe000a6fdf442ba2396391a3bec12644903`). One real file, `sample/10BT/000_00000.parquet`, produced 726,000 documents and 731,876,929 tokens in about 31 minutes. The agent confirmed that the shard size and EOS count match the summary. Next: rerun to confirm the completed shard is skipped, then commit and open the pull request. Task 009 (`PackedTokenDataset`, reading token files for training) follows in a separate pull request.

Current task: [007 — Load parquet datasets from the Hugging Face Hub](007-load-dataset.md), on branch `feat/load-dataset` after PR #1 merged. The user added `huggingface-hub` and `pyarrow`; `uv sync --locked` and the existing suite (`19 passed`) succeeded. Green on 2026-11-08: the user implemented `load_dataset` in `core/data.py`; the full suite returned `23 passed`. Next: a one-off manual check against a real FineWeb-Edu file, and confirming the exact Llama 3 tokenizer repo with the lab. Evaluation sets are scored by `flock eval` and are out of scope.

Latest completed task: [006 — Fit a small linear regression end to end](006-fit-linear-regression.md). A verification scenario, not a new feature: 30 `train(loader)` calls over `y = 3x + 1` reach weight approximately `3` and bias approximately `1`. Passed on first run with no production changes; no Red is claimed. A scratch `lr=0.0` copy failed as expected. Full suite in Docker: `19 passed` on 2026-11-08. Next minimum-viable gap: training loss is not observable from `train()` (Logger area).

Latest completed task: [005 — Minimal data loader](005-data-loader.md), Green. The user implemented `TensorDataset` and `DataLoader(dataset, batch_size=1, drop_last=False)` in `core/data.py`; `Trainer.train` consumes the loader unchanged. Full suite in Docker (`PYTHONPATH=src uv run pytest -q`): `18 passed` on 2026-11-08. Shuffle, custom collation, and invalid `batch_size` handling remain deferred.

Environment note: one contributor runs tests in a Linux Docker container because torch 2.15.0 has no Intel macOS wheel; `./test.sh` remains the runner for native `.venv` environments.

Latest task: [003 — Model forward hooks](003-model-forward-hooks.md), now Green. The current implementation calls `self.model(inputs)`. Verified on 2026-11-07: `./test.sh tests/unit/core/test_trainer_loss.py::test_compute_loss_uses_output_modified_by_model_forward_hook` returned exit 0, `1 passed in 2.10s`; `./test.sh` returned exit 0, `6 passed in 1.96s`. The hook-modified loss is accepted and previous scenarios remain Green. Production code was not modified by the agent.

Latest completed task: [004 — Reject empty batches](004-empty-batches.md), now Green. With explicit user authorization, the agent replaced the truthiness guard with a zero-step check after traversal. Empty lists and empty iterators now raise `ValueError` without parameter updates. `./test.sh tests/unit/core/test_trainer_empty_batches.py` returned exit 0, `2 passed in 2.03s`; `./test.sh` returned exit 0, `8 passed in 1.91s`. Existing nonempty iterator training remains Green. Next: agree on the next behavior before adding tests. Returned-state snapshot and error-state semantics remain deferred.

### Previous Task Verification

The user agreed on per-call progress and fixed the counter initialization: `train()` resets progress and initializes the local `steps` to integer zero. The agent did not modify production code.

Verified on 2026-11-07:

- `./test.sh tests/integration/test_minimal_training_loop.py::test_train_counts_steps_per_call_and_continues_updating_parameters`: exit 0, `1 passed in 2.02s`.
- `./test.sh`: exit 0, `5 passed in 2.01s`.
- Two successive one-batch calls return immediate `global_step` values `1 → 1`, with real SGD weights approximately `0.4 → 0.72`. Existing scalar-loss, training-mode, and fresh-gradient scenarios remain Green.
- The earlier `NoneType + int` counter regression is resolved. Its prior full-suite result was `3 failed, 2 passed in 1.85s`; the repeated-call test then failed before reaching its assertions.
- Empty-input, error-state, and returned-state snapshot semantics remain deferred. Model invocation and hook behavior are now covered by task 003 above.

The user normally writes production code. The agent supports design, tests, failure analysis, and documentation under [AGENTS.md](../AGENTS.md); the user explicitly authorized the task 004 production fix. No dependencies or test configuration were changed.

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
| [008](008-write-token-shard.md) | Exclude dev ids, tokenize, append EOS, write a `uint32` token shard. | Exact token sequence, counts, no partial file on failure. | Green (`26 passed`); manual real-data check: 1,000 documents, 1,037,837 tokens, exact decode round trip. Wiring (`load_dev_ids`, `prepare_token_shards` with resumable per-file shards): Green, `31 passed`. |
| [009](009-packed-token-dataset.md) | Read a `uint32` shard as packed `(x, y)` next-token samples for `DataLoader`. | Exact samples and shared boundary token, `len` cases, `int64` with large ids, tiny LM loss below `0.05`. | Green: `39 passed`; real shard yields 178,680 samples of 4,096 tokens. |

Custom operation kernels are not needed for this milestone. No Hugging Face differential compatibility requirement has been selected.

## Maintaining This Record

After meaningful changes, record the actual implementation, test commands and outcomes, blockers, and next action. Distinguish verified behavior from untested expectations and historical results.
