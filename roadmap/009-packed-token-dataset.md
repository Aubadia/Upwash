# 009 — Read a Token Shard as Packed Training Samples

## Behavior

Connect the token shards from task 008 to the existing `DataLoader` and `Trainer`. Read a `.bin` shard without loading it into memory and split its token stream into fixed-length samples. Each sample pairs `seq_len` input tokens with the next-token labels. This runs at training time, once per sample; task 008 runs once, ahead of training.

```text
.bin shard → PackedTokenDataset → DataLoader → Trainer.train
```

Branch `feat/packed-token-dataset`, created from `feat/load-dataset`. This will be a separate pull request: the two tasks share only the shard format (raw `uint32` ids, documents joined with EOS), and both edit `core/data.py`.

## Agreed Interface

```python
from upwash.core.data import PackedTokenDataset

dataset = PackedTokenDataset(path, seq_len=4096)
len(dataset)        # (num_tokens - 1) // seq_len
x, y = dataset[i]   # int64 tensors of shape [seq_len]
```

- Sample `i` reads tokens `[i * seq_len, i * seq_len + seq_len + 1)`. `x` is the first `seq_len` tokens; `y` is the last `seq_len`, shifted by one. Neighbouring samples share one boundary token.
- Leftover tokens that cannot form a full sample are dropped.
- Packing: samples cross document boundaries and use no padding. EOS tokens in the stream mark document ends.
- The shard is opened read-only through `np.memmap`, so only the requested slice is read from disk.
- Returned tensors are `int64`, as embedding lookups and cross-entropy targets require.
- Placement in `core/data.py`, like the other data classes.

Deferred: multiple shards, shuffling, index bounds checks, attention masks at document boundaries, a token budget, resume position, and distributed sharding.

## Tests

`tests/unit/core/test_packed_token_dataset.py` writes small `uint32` files with numpy:

- Tokens `10..18`, `seq_len=4`: `dataset[0] == ([10, 11, 12, 13], [11, 12, 13, 14])` and `dataset[1] == ([14, 15, 16, 17], [15, 16, 17, 18])`.
- `len` for 9, 10, 8, 5, and 4 tokens with `seq_len=4`: `2, 2, 1, 1, 0`.
- `int64` dtype, with `128001` (above `uint16`) preserved.

`tests/integration/test_train_language_model_on_packed_tokens.py`:

- A repeating `0..7` stream (65 tokens) goes through `DataLoader(batch_size=4)`, producing `[4, 4]` `int64` batches.
- A tiny embedding plus linear model is trained with `Trainer`, Adam (`lr=0.1`), and a test-supplied next-token cross-entropy, for 30 `train` calls.
- `global_step == 4` per call; the mean loss starts above `1.0` (uniform guessing costs `ln 8 ≈ 2.08`) and ends below `0.05`.

## Verified Red — 2026-11-08

- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_packed_token_dataset.py tests/integration/test_train_language_model_on_packed_tokens.py`.
- Result: `ImportError: cannot import name 'PackedTokenDataset' from 'upwash.core.data'` in both files; collection errors. Missing API only. Rest of the suite: `31 passed`.
- Test sanity check: a scratch reference implementation outside the repository passed all eight tests (`8 passed`).
- Margin check: across seeds 0–4, the final mean loss was below `1.2e-6` (initial `2.1`–`2.6`), so the `0.05` threshold is not seed-sensitive.

## Verified Green — 2026-11-08

- The user implemented `PackedTokenDataset` in `core/data.py`:
  - `__init__` stores `seq_len` and a read-only `np.memmap` of `uint32`;
  - `__len__` returns `(len(self.tokens) - 1) // self.seq_len`;
  - `__getitem__` slices `seq_len + 1` tokens from `i * seq_len`, converts them to `int64` with `torch.from_numpy`, and returns `chunk[:-1], chunk[1:]`.
- Command (Docker): the two 009 test files returned `8 passed`. Full suite: `39 passed`.
- Red iterations during implementation:
  - The memmap was not stored (`AttributeError: ... no attribute 'tokens'`).
  - Other defects were fixed before tests reached them: `len(self)` used in place of `seq_len` (infinite recursion in `__len__`), a `self.token` typo, and `chunk[:1]` in place of `chunk[1:]`.

## Real-Shard Smoke Check — 2026-11-08

An agent-run read-only check on `.cache/tokens/sample/10BT/000_00000.bin` from task 008, with `seq_len=4096`:

- `len` is 178,680, which equals `(731,876,929 - 1) // 4096`.
- The last sample has shape `(4096,)`, dtype `int64`, and the labels are the inputs shifted by one.
- `DataLoader(batch_size=8)` yields an `(8, 4096)` batch.
- Opening the 2.9 GB shard and reading these samples took well under a second, so the memmap does not load the file.

## Explicit Byte Order — 2026-11-09

This follows the task 008 review, which made the shard format explicit little-endian.

- The branch was rebased onto the updated `feat/load-dataset` (`9c3fe8c`) without conflicts.
- The user changed the memmap to `dtype="<u4"`, and the tests now write their shards with `"<u4"`.
- Full suite (Docker): `40 passed`.
- No Red is claimed: on little-endian hardware, native `uint32` and `"<u4"` are identical.

## Merge with `main` — 2026-11-09

- PR #4 (configuration, MemoryLogger, Builder; tasks 010–012) merged into `main`. It renumbered its tasks to 010–014, which leaves 009 for this task.
- `main` was merged into this branch. The only conflict was `roadmap/README.md`. It was resolved from `main`'s version, with this task added.
- Full suite on the merged tree (Docker): `54 passed`.

## Review: Argument and Index Checks — 2026-11-09

Review on PR #3 asked for validation of `seq_len` (line 59) and of the index (line 64). This brings the deferred index bounds checks into this task. Agreed behavior, following Python sequences:

- `seq_len <= 0` raises `ValueError` in the constructor, before the shard is opened.
- `dataset[-k]` equals `dataset[len - k]` for `1 <= k <= len`.
- An index `>= len` or `< -len` raises `IndexError`.
- Plain iteration (`iter(dataset)`, which indexes until `IndexError`) yields exactly `len` samples.

Tests added to `tests/unit/core/test_packed_token_dataset.py`. Iteration is capped with `islice`, so a missing `IndexError` fails the test instead of hanging it.

Verified Red (Docker, `uv run pytest -q tests/unit/core/test_packed_token_dataset.py`): `8 failed, 7 passed`. Full suite: `8 failed, 54 passed`. Causes, all from the missing checks:

- `seq_len` 0 and -1: `DID NOT RAISE ValueError`. The constructor does no arithmetic with `seq_len`.
- `dataset[-1]` returns empty tensors: `start = -4`, so the slice `[-4:1]` is empty.
- `dataset[-2]` returns the wrong tokens, `[11, 12, 13, 14]`: the slice `[-8:-3]` counts from the end of the file.
- `dataset[2]`, `[3]`, `[-3]`: `DID NOT RAISE IndexError`.
- Iteration over 10 tokens yields 3 samples; the third is `([18], [19])`, built from the leftover token.

Verified Green — 2026-11-09. The user implemented both checks:

- `__init__` raises `ValueError(f"seq_len must be positive, got {seq_len}")`.
- `__getitem__` adds `len(self)` to a negative index, then raises `IndexError` if the index is still outside `[0, len)`.

Command (Docker): `uv run pytest -q tests/unit/core/test_packed_token_dataset.py` returned `15 passed`. Full suite: `62 passed`.

Refinements suggested to the user; no test covers either:

- The check runs after `np.memmap`, so an invalid `seq_len` still opens the shard. The agreed order is to check first.
- The `IndexError` message reports the converted index (`-1` for `dataset[-3]`) instead of the caller's value.

Status: complete for single-shard, in-order training data. The deferred items above are next for the data area. A real model, optimizer schedule, and device handling are outside this task.
