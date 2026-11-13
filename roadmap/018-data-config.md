# 018 — Data Configuration and `build_dataset`

## Behavior

Describe a run's training data in configuration and build the dataset from it. This connects tasks 015 and 016 (`ConcatDataset`, `Truncate`, `PackedTokenDataset.from_npy`) to configuration. Each scale then needs only a different `num_tokens`.

```text
DataConfig → build_dataset → sorted(root.glob(pattern)) → PackedTokenDataset (per file) → ConcatDataset → Truncate(num_tokens // seq_len)
```

Branch `feat/optimizer-schedule-config`, as a second commit after task 017 (`7a8ed2c`), in the same pull request.

## Agreed Interface — 2026-11-13

```python
from upwash.builder import build_dataset
from upwash.config import DataConfig

config = DataConfig(
    root="/data/fineweb-edu-llama3",
    pattern="*/tokens.npy",
    format="npy",                 # "npy" (Flock) or "bin" (task 008 shards)
    seq_len=4096,
    num_tokens=30_000_000_000,    # 150M at 200 tokens per parameter
)
dataset = build_dataset(config)
```

`DataConfig` holds settings only and doesn't touch the file system:

- Every field is required, as in task 017. A missing field raises `TypeError` naming it.
- `format` must be `"npy"` or `"bin"`. `seq_len` must be greater than 0. `num_tokens` must be at least `seq_len`, so the run has at least one sample. Violations raise `ValueError`.
- `to_dict()` returns every field.

`build_dataset(config)` lives in `builder.py`:

- Files are `sorted(Path(root).glob(pattern))`, so the order is the same on every machine. `"*/tokens.npy"` skips the `offsets.npy` next to each Flock `tokens.npy`.
- `"npy"` opens each file with `PackedTokenDataset.from_npy`. `"bin"` opens it with `PackedTokenDataset(path, seq_len)`.
- No matching file raises `ValueError`. A budget larger than the data raises `ValueError` (from `Truncate`).
- The budget is `num_tokens // seq_len` samples. A partial sample is dropped, so a run never exceeds its token budget.

The budget is in tokens because the experiment plan states budgets in tokens (30B, 60B, 100B, 150B). Token counts are given directly rather than computed from a parameter count, because "150M" depends on how parameters are counted.

Deferred:

- Seeded shuffle, per-rank sharding, and resume position.
- Checking that `seq_len` matches the model's context length, in a top-level experiment config.
- Mixing formats in one run.
- `Path` values for `root`.

## Tests

`tests/unit/test_data_config.py`:

- Fields are exposed, and each missing field raises `TypeError` naming it.
- A nonexistent `root` is accepted at construction.
- `"npy"` and `"bin"` are accepted. `"NPY"`, `"parquet"`, `"u4"`, and `""` are rejected.
- `seq_len` 0 and -1 are rejected.
- `num_tokens == seq_len` is accepted. 4095 (for `seq_len` 4096), 0, and -1 are rejected.
- `to_dict()` returns every field.

`tests/unit/test_build_dataset.py`, with `seq_len=4`:

- Two Flock-style folders are written out of order: `000001` with 5 tokens and `000000` with 10 tokens, each with an `offsets.npy`. A 12-token budget gives 3 samples. Sample 0 is `[0..3]`, and sample 2 is `[100..103]`.
- Budgets of 8, 11, and 4 tokens give 2, 2, and 1 samples. The last sample is checked.
- A 16-token budget needs 4 samples, which raises `ValueError`. A pattern matching nothing raises `ValueError`.
- Two `.bin` shards under `sample/10BT/` are read in sorted order.
- `DataLoader(batch_size=2)` gives shapes `(2, 4)` and `(1, 4)`.

## Verified Red — 2026-11-13

- Command (Docker, `upwash-venv-pr5`): `PYTHONPATH=src uv run pytest -q tests/unit/test_data_config.py tests/unit/test_build_dataset.py`.
- Result: two collection errors. One is `ImportError: cannot import name 'DataConfig' from 'upwash.config'`. The other is `ImportError: cannot import name 'build_dataset' from 'upwash.builder'`. Missing API only.
- Rest of the suite, ignoring the two files: `160 passed`.
- Test sanity check: a scratch reference implementation outside the repository passed both files (`28 passed`). It used a dataclass plus `sorted(glob)` → `PackedTokenDataset`/`from_npy` → `ConcatDataset` → `Truncate`, loaded as a pytest plugin.

## Verified Green — 2026-11-13

The user implemented the following:

- `DataConfig` is a dataclass. `__post_init__` checks `format`, `seq_len > 0`, and `num_tokens >= seq_len`, and `to_dict()` returns `asdict`.
- `build_dataset` does `sorted(Path(root).glob(pattern))`, raises `ValueError` when nothing matches, opens each file with `from_npy` or `PackedTokenDataset` depending on `format`, and returns `Truncate(ConcatDataset(shards), num_samples=num_tokens // seq_len)`.

Command (Docker, `upwash-venv-pr5`): the two files returned `28 passed`. Full suite: `188 passed`.

Red iterations during implementation:

- `DataConfig` error messages referenced `self.decay_shape`, which was copied from `ScheduleConfig`. Building the message raised `AttributeError` before the intended `ValueError`, and 10 cases failed.
- `num_tokens > seq_len` rejected a budget of exactly one sample.
- At the user's explicit request, the agent fixed those two defects in `config.py` (messages and `>=`).
- `build_dataset` then passed `num_tokens` as `num_samples`, without `// seq_len`. Six cases failed with `ValueError: num_samples must be in [0, 3], got 12`. The over-budget case passed only by coincidence before the fix.

Real-file smoke check (agent-run, read-only, scratch copy of Flock folder `000011` mounted as `/data/000011`, `pattern="*/tokens.npy"`):

- A 30M-token budget gives 7,324 samples (`30_000_000 // 4096`). Building took 0.003 s. The first batch is `(8, 4096)` `int64` and starts with BOS 128000.
- A 30B budget on that single folder raises `ValueError: num_samples must be in [0, 15395], got 7324218`.

Status: complete. Next: commit as the second commit of this pull request.
