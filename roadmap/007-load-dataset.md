# 007 — Load Parquet Datasets from the Hugging Face Hub

## Behavior

Provide `load_dataset` so training data can be read from a pinned Hugging Face dataset, reusing a local copy when one already exists. This is the first step of the FineWeb-Edu preprocessing pipeline (download → exclude dev ids → tokenize → token files). Tokenization, dev-id exclusion, and token files are later tasks.

Scope decisions agreed with the user and teammate on 2026-11-08:

- Generic over datasets: callers choose the dataset through parameters, with no per-dataset code.
- Parquet only. Every dataset this lab needs is parquet: training data `HuggingFaceFW/fineweb-edu` (`sample/10BT`, `sample/100BT`, `sample/350BT`), and the eval sets read by Flock. Other formats are deferred until required.
- Evaluation sets (`fineweb_edu`, `base_easy`, `paloma`) are scored by `flock eval`, which loads them itself; they are not a requirement for this task.
- Dev-id exclusion stays outside `load_dataset` so it remains generic.
- Dependencies `huggingface-hub` (2.4.0) and `pyarrow` (26.0.1) were added by the user.

## Proposed Interface

```python
from upwash.core.data import load_dataset

rows = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    revision="<commit>",
    data_files="sample/10BT/*.parquet",
    columns=["id", "text"],
    cache_dir=None,
    download=huggingface_hub.snapshot_download,
)
for row in rows:  # {"id": ..., "text": ...}
    ...
```

- `download` defaults to `huggingface_hub.snapshot_download` and is called as `download(repo_id, repo_type="dataset", revision=..., allow_patterns=data_files, cache_dir=...)`, returning the local snapshot folder. The downloader handles caching: a pinned revision already on disk is not downloaded again. The parameter lets tests run offline; the user may change this.
- Files under the snapshot matching `data_files` are read in sorted path order, so row order is deterministic.
- Only the requested columns are read, in batches, and yielded one row at a time as dicts.
- Placement in `core/data.py` is provisional and can move.

Deferred, pending agreed behavior: errors when no file matches or a column is missing; formats other than parquet.

## Tests

`tests/unit/core/test_load_dataset.py` writes small parquet files under pytest's `tmp_path` and passes a fake downloader, so no network access is needed:

- Selected columns only, only files matching the pattern (a `sample/100BT` file is ignored), sorted file order even when files are created in reverse.
- The downloader receives the repo id, `repo_type="dataset"`, the pinned revision, the file pattern as `allow_patterns`, and `cache_dir`.

## Verified Red — 2026-11-08

- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_load_dataset.py`.
- Result: `ImportError: cannot import name 'load_dataset' from 'upwash.core.data'`; collection error. This confirms the missing API only.
- Test sanity check: a scratch reference implementation outside the repository passed both tests (`2 passed`). The repository contains no implementation.
- Before adding the tests, the full suite passed (`19 passed`) after `uv sync --locked` with the new dependencies.

## Progress — 2026-11-08

- The user implemented `load_dataset` as a generator: download, `sorted(Path(local_dir).glob(data_files))`, `pq.ParquetFile(path).iter_batches(...)`, and a row-by-row `yield`. The full suite passed (`21 passed`).
- The implementation passed `columns=["id", "text"]` literally rather than the `columns` argument; both original tests happened to request those columns. Added `test_load_dataset_reads_the_columns_the_caller_requests` (`columns=["text", "score"]`). Red confirmed: `1 failed, 2 passed`; observed `{'id': 'a', 'text': 'first'}`, expected `{'text': 'first', 'score': 0.1}`.

- Agreed error behavior: when no file matches `data_files`, raise `ValueError` with the pattern in the message. Missing-column handling is deferred by user decision (pyarrow's own error applies). Added `test_load_dataset_raises_when_no_file_matches`; Red confirmed: `Failed: DID NOT RAISE ValueError`.
- Real-data check (manual, network): the `HuggingFaceFW/fineweb-edu` main revision on 2026-11-08 is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- Tokenizer: the lab's vocabulary of 128,256 matches the Llama 3 tokenizer (128,000 regular tokens plus 256 special tokens). `meta-llama/Llama-3.2-1B` is gated with manual approval on the Hugging Face Hub. The exact tokenizer repo must be confirmed with the lab; Flock eval loads the tokenizer from the submitted model repo (or the Flock model config's `hf_repo`).

## Verified Green — 2026-11-08

- The user passed the `columns` argument to `iter_batches` and raised `ValueError(f"no files match {data_files!r} in {local_dir}")` when nothing matches.
- Full suite in Docker: `PYTHONPATH=src uv run pytest -q` returned `23 passed` (four `load_dataset` tests).

## Manual Checks — 2026-11-08

- Real data: `load_dataset("HuggingFaceFW/fineweb-edu", revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", data_files="sample/10BT/000_00000.parquet", columns=["id", "text"], cache_dir=".cache/huggingface")` downloaded the file into the project's ignored cache. The cached file is 2,152,819,114 bytes, matching the size listed on the Hub. The user ran the read of the first rows.
- Tokenizer access: with the user's own `HF_TOKEN` passed into Docker, `meta-llama/Llama-3.2-1B` (gated) downloaded `tokenizer.json` and `tokenizer_config.json` at revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. Vocabulary `128000 + 256 = 128256`, matching the lab; BOS `<|begin_of_text|>`, EOS `<|end_of_text|>` (base-model settings). The lab has not yet confirmed this exact repo and revision.

## Next Action (Superseded)

Run the one-off manual check against a real file (network, about 2.15 GB): `sample/10BT/000_00000.parquet` at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, with `cache_dir` under the project's ignored `.cache/` so the Docker container keeps the download. Confirm the exact tokenizer repo with the lab before starting tokenization.
