# 005 — Minimal Data Loader

## Behavior

Provide a minimal map-style dataset and a data loader in `src/upwash/core/data.py` that `Trainer.train` can consume as an iterable of `(inputs, targets)` batches. Reading files, decoding, and tokenization are out of scope: as with Hugging Face `Trainer`, the dataset arrives already prepared, and the loader owns ordering, batching, and collation.

## Proposed Interface (Pending User Confirmation)

Following the user's request to use the most common parameters, the tests use names matching `torch.utils.data`:

```python
from upwash.core.data import DataLoader, TensorDataset

dataset = TensorDataset(inputs, targets)        # len(dataset); dataset[i] -> (inputs[i], targets[i])
loader = DataLoader(dataset, batch_size=1, drop_last=False)
state = trainer.train(loader)
```

- Samples are stacked along a new leading dimension: samples of shape `(1,)` form batches of shape `(batch, 1)`.
- Batches follow dataset order. The final partial batch is kept unless `drop_last=True`.
- `len(loader)` returns the batch count: ceiling division by default, floor division with `drop_last=True`.
- The loader can be iterated more than once, yielding the same batches each time.

Deferred: `shuffle` and seeding, custom `collate_fn`, dict batches and padding, empty-dataset and invalid-`batch_size` behavior, workers, and pinned memory. The user may revise names or defaults; tests will be updated only after agreement.

## Tests

- `tests/unit/core/test_data.py`: dataset length and pairing; batch order and partial batch; `drop_last`; default `batch_size=1`; `len`; repeated iteration.
- `tests/integration/test_training_with_data_loader.py`: `Trainer.train(DataLoader(...))` over two identical samples with `batch_size=1` gives weight approximately `0.72` and `global_step == 2`.

Five samples use inputs `0..4` and targets `10 × inputs` so ordering and pairing errors are visible. This is not a differential comparison; `torch.utils.data` was used only as a one-off sanity check of the tests.

## Verified Red — 2026-11-08

- Command (Docker, the user's Intel Mac environment): `PYTHONPATH=src uv run pytest -q`.
- Result: `2 errors during collection`, `ImportError: cannot import name 'DataLoader' from 'upwash.core.data'`. `core/data.py` is empty.
- This confirms the missing API only; no batching behavior has been exercised against Upwash. The collection error interrupts the whole run, so existing tests also do not execute until the classes exist.
- Test sanity check: the same tests with the import swapped to `torch.utils.data` gave `8 passed` (run from a scratch copy; repository unchanged).

## Progress — 2026-11-08

- `TensorDataset` Green: `uv run pytest tests/unit/core/test_data.py -k tensor_dataset` passed.
- `DataLoader.__len__`: the original 5-sample cases passed, but the implementation adds `1` to floor division whenever `drop_last=False`. Added divisible cases (4 samples, `batch_size=2`). Red confirmed: `divisible-keep` failed with `assert 3 == 2`; `1 failed, 3 passed`.
- The user fixed `__len__` with ceiling division `(n + batch_size - 1) // batch_size`; all four `len` cases passed.

## Verified Green — 2026-11-08

- The user implemented `DataLoader.__iter__` as a generator: the outer loop runs `len(self)` batches, the inner loop collects samples from `start` to `min(start + batch_size, len(dataset))`, and each batch is collated with `torch.stack`.
- Full suite in Docker: `PYTHONPATH=src uv run pytest -q` returned `16 passed`, including the Trainer integration test (weight approximately `0.72`, `global_step == 2`).
- Renumbered from 004 to 005 after syncing with `origin/main`, which added [004 — Reject empty batches](004-empty-batches.md) and moved `TrainingState` to `core/state.py`. After the sync, the full suite returned `18 passed` (8 existing Trainer tests and 10 data tests).
- The interface in this file was used as implemented by the user; it is treated as agreed for this task.
- Suggested user-owned cleanup: rename `batch_targers` to `batch_targets`; return `len(self.inputs)` directly in `TensorDataset.__len__`.

## Next Action

Agree on the next data behavior: invalid `batch_size` (currently `0` raises `ZeroDivisionError` from `__len__`), empty datasets, or `shuffle` with explicit seeding. Write its test after the expected behavior is agreed.
