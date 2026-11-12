# 015 — Read Many Shards as One Dataset, with a Sample Budget

## Behavior

Task 009 reads one shard. A 150M run at 200 tokens per parameter needs about 30B tokens, or about 41 shards of the size task 008 produced (about 0.73B tokens each). This task joins many shards into one map-style dataset. It also stops at a fixed number of samples, so a run trains on exactly its token budget.

```text
.bin shards → PackedTokenDataset (one per shard) → ConcatDataset → Truncate → DataLoader → Trainer.train
```

This is not parallel reading. The model still consumes one batch at a time. One index over all shards is also what a later seeded shuffle and per-rank sharding will permute and split.

Branch `feat/multi-shard-dataset`, created from `main` at `7e60792`.

## Agreed Interface — 2026-11-12

```python
from upwash.core.data import ConcatDataset, PackedTokenDataset, Truncate

shards = [PackedTokenDataset(p, seq_len=4096) for p in sorted(paths)]
dataset = ConcatDataset(shards)
dataset = Truncate(dataset, num_samples=budget_tokens // seq_len)
```

`ConcatDataset(datasets)`:

- `len` is the sum of the parts' lengths.
- Global index `i` maps, in order, to a part and a local index inside it. Empty parts are skipped.
- Negative indices count from the end, and indices outside `[-len, len)` raise `IndexError`, as in `PackedTokenDataset`.
- The parts are any map-style datasets (`len` and `[i]`). Samples never cross shard boundaries, because each part is a separate `PackedTokenDataset`. Each shard drops fewer than `seq_len` leftover tokens.

`Truncate(dataset, num_samples)`:

- Keeps the first `num_samples` samples, in order. `len` is `num_samples`.
- Raises `ValueError` when `num_samples` exceeds `len(dataset)`, so a run can't silently train on less data than its budget. It also raises when `num_samples` is negative. That case was added in the tests, and the user should confirm it.
- Negative indices count from the truncated end (`dataset[-1]` is the last kept sample). Indices outside `[-num_samples, num_samples)` raise `IndexError`.
- The budget is in samples, not tokens. Each sample trains on `seq_len` tokens, so the caller computes `budget_tokens // seq_len`, and `Truncate` doesn't need to know `seq_len`.

Both classes go in `core/data.py`.

Deferred: seeded shuffle, and whether shuffling happens before or after truncation. Also deferred: resume position, per-rank sharding, an empty list of parts (`ConcatDataset([])`), non-integer indices, and preprocessing speed.

## Tests

`tests/unit/core/test_concat_dataset.py` uses a strict list-backed test dataset. It fails on any local index outside `[0, len)`, so a wrong translation fails loudly:

- Parts of lengths 2, 1, 3: `len == 6`, and indices `0..5` give `a0 a1 b0 c0 c1 c2`.
- An empty middle part is skipped.
- Negative indices `-1`, `-4`, `-6` give `c2`, `b0`, `a0`. Indices `6`, `7`, `-7` raise `IndexError`.
- Plain iteration yields exactly `len` items. It is capped with `islice`, so a missing `IndexError` fails instead of hanging.
- Two real `"<u4"` shards (10 tokens and 5 tokens, `seq_len=4`):
  - `len == 3`;
  - the second shard's first sample starts at its own first token (`[100..103]`), not at the first shard's leftover token;
  - `DataLoader(batch_size=2)` gives batches of shape `(2, 4)` and `(1, 4)`.

`tests/unit/core/test_truncate.py`, over a source of six items:

- `num_samples=4` keeps `s0..s3`. `num_samples=6` keeps everything.
- `num_samples=7` and `-1` raise `ValueError`.
- Negative indices `-1` and `-4` give `s3` and `s0`. Indices `4`, `5` (present in the source, but outside the budget), and `-5` raise `IndexError`.
- Iteration stops after four items.

## Verified Red — 2026-11-12

- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_concat_dataset.py tests/unit/core/test_truncate.py`.
- Result: `ImportError: cannot import name 'Truncate' from 'upwash.core.data'`, with collection errors in both files (`2 errors`). Missing API only.
- Rest of the suite, ignoring the two new files: `62 passed`.
- Test sanity check: a scratch reference implementation outside the repository passed all new tests (`21 passed`). It uses cumulative part lengths with `bisect`.

## Verified Green — 2026-11-12

The user implemented both classes in `core/data.py`:

- `ConcatDataset.__init__` stores `list(datasets)` and records cumulative lengths (`self.ends`). `__len__` returns `self.ends[-1]`. `__getitem__` first normalizes a negative index and checks the range. It then finds the part with `bisect.bisect_right(self.ends, i)` and reads `self.datasets[k][i - start]`, where `start` is the previous part's end, or `0` for the first part.
- `Truncate.__init__` raises `ValueError` when `num_samples` is outside `[0, len(dataset)]`, then stores both arguments. `__len__` returns `num_samples`. `__getitem__` normalizes and checks the index against `num_samples`, then returns `self.dataset[i]`.

Command (Docker): `uv run pytest -q tests/unit/core/test_concat_dataset.py tests/unit/core/test_truncate.py` returned `21 passed`. Full suite: `83 passed`.

Red iterations during implementation:

- `__getitem__` read `self.datasets[k][start - i]`. The strict test dataset rejected the negative local index (`AssertionError: local index -1 outside [0, 2)`, four tests).
- `Truncate` had `getitem` in place of `__getitem__` and no `__len__` (`TypeError: 'Truncate' object is not subscriptable` and `has no len()`). Then `__len__(self, dataset)` took an extra parameter (`TypeError: ... missing 1 required positional argument`, three tests).
- One defect no test covers was fixed in review: the length loop iterated the `datasets` argument instead of the stored list. A generator argument would have been consumed by `list()`, leaving `ends` empty.

Real-shard smoke check (agent-run, read-only, on the task 008 shard concatenated with itself, `seq_len=4096`):

- `len` is 357,360, twice the single shard's 178,680. Index 178,680 returns the second copy's first sample.
- `Truncate` to a 30M-token budget gives 7,324 samples. Its last sample equals the concatenation's sample 7,323.
- `DataLoader(batch_size=8)` yields an `(8, 4096)` `int64` batch.
- The check took 0.01 s.

Open refinement, not covered by tests: like task 009, the `IndexError` messages report the normalized index instead of the caller's value.

Status: complete for ordered multi-shard reading with a sample budget. Next for the data area: seeded shuffle, including whether it applies before or after `Truncate`.
