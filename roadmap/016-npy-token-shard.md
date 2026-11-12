# 016 — Read Flock's Pre-Tokenized `.npy` Shards

## Behavior

Flock hosts FineWeb-Edu already tokenized with the Llama 3 tokenizer, at `cerulean-labs/scaling-llama-3-2/fineweb-edu-llama3/NNNNNN/{tokens.npy,offsets.npy}`. Reading it saves preprocessing time for experiments. Our own preprocessing (tasks 007–008) stays as verification and backup.

```text
Flock S3 → sync to local disk (nano4) → PackedTokenDataset.from_npy (one per folder) → ConcatDataset → Truncate → DataLoader → Trainer.train
```

Format, checked read-only on 2026-11-12 against folder `000011`:

- `tokens.npy` is a 1-D `int32` array with a `.npy` header, 63,061,856 tokens, ids 0..128001.
- Documents are stored as `[BOS, text, EOS]`. BOS (128000) and EOS (128001) both occur 64,168 times, and every EOS is followed by a BOS.
- 50 of 50 sampled documents re-tokenize identically with our tokenizer.
- Not verified: dev-set exclusion (the file has no document ids) and the tokenizer revision (not pinned in `meta.json`). The user believes the dev set is excluded.

Because of the header, `np.memmap(path, dtype=...)` would read the header bytes as tokens. `np.load(path, mmap_mode="r", allow_pickle=False)` parses the header and memory-maps the data. Memory-mapping needs a local file. Another group read `.npy` files directly from S3 with `np.load`, which can't memory-map a remote object, and reported a 42-hour 150M run bottlenecked on S3 reads. So the data is synced to local disk first.

Our `.bin` shards have no BOS, so one scaling ladder must use a single data source. `ConcatDataset` doesn't check this.

Branch `feat/multi-shard-dataset`, in the same pull request as task 015.

## Agreed Interface — 2026-11-12

```python
from upwash.core.data import PackedTokenDataset

dataset = PackedTokenDataset.from_npy("000011/tokens.npy", seq_len=4096)
```

- A classmethod on the existing class. It returns a `PackedTokenDataset`. `PackedTokenDataset(path, seq_len)` for `.bin` shards is unchanged.
- Opens the file with `np.load(path, mmap_mode="r", allow_pickle=False)`.
- Otherwise it behaves exactly like the `.bin` form: `len == (num_tokens - 1) // seq_len`, `int64` `(x, y)` pairs, negative indices, `IndexError` out of range, and `ValueError` for `seq_len <= 0`.
- Tokens are used as stored. BOS isn't removed or added.

Rejected alternatives: converting `.npy` to `.bin` (an extra copy of about 120 GB for 30B tokens), choosing the format from the file suffix (implicit), and changing `__init__` to take an array (breaks the 009 API and tests).

Deferred: `offsets.npy`, and errors for non-1-D arrays, non-integer dtypes, and pickled object arrays.

## Tests

Appended to `tests/unit/core/test_packed_token_dataset.py`. Files are written with `np.save` as `int32`, like Flock's:

- The same 14 tokens saved as `.bin` and as `.npy` give the same `len` (3) and the same samples. The result is a `PackedTokenDataset`.
- `[128000, 7, 128001, 128000, 9]` gives `int64` tensors with the ids unchanged.
- `seq_len` `0` and `-1` raise `ValueError`.
- `dataset[-1]` is the last sample. Indices `2` and `-3` raise `IndexError` (`len` 2).
- Two `.npy` shards in `ConcatDataset`: `len` 3, the second shard's sample starts at its own first token, and `DataLoader(batch_size=2)` gives shapes `(2, 4)` and `(1, 4)`.

The tests don't check memory-mapping, because that is an implementation detail. A read-only check on the real Flock file after Green will confirm it.

## Verified Red — 2026-11-12

- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_packed_token_dataset.py`.
- Result: `6 failed, 15 passed`. All six new cases fail with `AttributeError: type object 'PackedTokenDataset' has no attribute 'from_npy'`. That is the missing API only, including the `ValueError` cases.
- Full suite: `6 failed, 83 passed`. No existing test regressed.
- Test sanity check: a scratch reference implementation outside the repository passed the file (`21 passed`). It was loaded as a pytest plugin.

## Verified Green — 2026-11-12

The user refactored `PackedTokenDataset`:

- `_setup(tokens, seq_len)` raises `ValueError` for `seq_len <= 0`, then stores `self.tokens` and `self.seq_len`.
- `__init__` opens the `.bin` shard with `np.memmap(path, dtype="<u4", mode="r")` and calls `_setup`.
- `from_npy` creates the object with `cls.__new__(cls)`, opens the file with `np.load(path, mmap_mode="r", allow_pickle=False)`, calls `_setup`, and returns the object.
- `__len__` and `__getitem__` are unchanged.

Command (Docker): `uv run pytest -q tests/unit/core/test_packed_token_dataset.py` returned `21 passed`. Full suite: `89 passed`.

One Red iteration during implementation gave `21 failed` in the file and `23 failed, 66 passed` in the full suite. There were two causes:

- `__init__` opened the shard but didn't call `_setup`, so `.bin` datasets had no `tokens` (`AttributeError`). The `.bin` `seq_len` tests also failed with `DID NOT RAISE ValueError`.
- `_setup` checked `self.seq_len` before assigning it, so every `from_npy` case failed with `AttributeError: ... no attribute 'seq_len'`.

Real-file check (agent-run, read-only, on a scratch copy of Flock folder `000011/tokens.npy`, `seq_len=4096`):

- Opening took 0.0045 s and gives a `memmap` of `int32`, shape `(63061856,)`. The file is memory-mapped, not loaded.
- `len` is 15,395, equal to `(63,061,856 - 1) // 4096`.
- Sample 0 starts with BOS 128000 and is `int64`. `y` is `x` shifted by one. The last sample is full length, with ids at most 128001.
- `Truncate(ConcatDataset([ds, ds]), 20000)` with `DataLoader(batch_size=8)` yields an `(8, 4096)` `int64` batch.

Status: complete for reading Flock `.npy` shards. Still open: verifying dev-set exclusion and the tokenizer revision of the Flock data, and syncing it to nano4.
