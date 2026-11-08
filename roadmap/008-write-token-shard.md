# 008 — Tokenize Documents into a Token Shard

## Behavior

Turn documents from `load_dataset` into the token file that training reads: skip the FineWeb-Edu dev set, tokenize each document without special tokens, append one EOS per document, and write the concatenated ids as `uint32`. This is the second step of the preprocessing pipeline (download → **exclude dev ids → tokenize → token file**). Reading the token file for training is task 009.

The user chose to continue this task in the task 007 pull request because it builds on `load_dataset`.

## Agreed Interface

```python
from upwash.core.data import write_token_shard

metadata = write_token_shard(
    rows,                 # iterable of {"id": ..., "text": ...}, e.g. load_dataset(...)
    tokenize=tokenize,    # callable: text -> list[int], without special tokens
    eos_id=eos_id,        # int, read from the tokenizer
    exclude_ids=dev_ids,  # set of document ids to skip
    output_path=path,     # destination .bin file
)
# metadata == {"num_docs": ..., "num_excluded": ..., "num_tokens": ...}
```

- Token ids are written as little-endian `uint32` (4 bytes each). Llama 3 ids reach 128,255, above `uint16`'s maximum.
- `num_tokens` includes EOS tokens; it is the quantity the 200B training budget counts.
- A failure while writing must not leave a file at `output_path`; write to a temporary path and rename on success.
- `tokenize` and `eos_id` are injected so tests run without downloading the tokenizer.
- Placement in `core/data.py` is provisional.

Deferred: the real tokenizer wrapper (needs the `tokenizers` dependency), reading `dev-ids.txt`, empty-document handling, a metadata file on disk, and parallel processing of many parquet files.

## Tokenizer Under Consideration

`meta-llama/Llama-3.2-1B` at revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`: vocabulary 128,256; BOS `<|begin_of_text|>`; EOS `<|end_of_text|>` (verified in task 007). Lab confirmation is pending and is needed before large-scale tokenization, not for these tests.

## Tests

`tests/unit/core/test_write_token_shard.py` uses a fake tokenizer (one token per character, its code point) and `EOS_ID = 128001`:

- Rows `a: "hi"`, `dev-1: "xx"` (excluded), `b: "yo!"` produce exactly `[104, 105, 128001, 121, 111, 33, 128001]`, 28 bytes, read back as `uint32`.
- Returned metadata `{"num_docs": 2, "num_excluded": 1, "num_tokens": 7}`.
- When `tokenize` raises partway through, the error propagates and no file exists at `output_path`.

## Verified Red — 2026-11-08

- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_write_token_shard.py`.
- Result: `ImportError: cannot import name 'write_token_shard' from 'upwash.core.data'`; collection error. Missing API only.
- Test sanity check: a scratch reference implementation outside the repository passed all three tests (`3 passed`). The repository contains no implementation.

## Verified Green — 2026-11-08

- The user implemented `write_token_shard` in `core/data.py`: it writes to `<output_path>.tmp`, skips ids in `exclude_ids`, appends `eos_id` after tokenizing each document, writes each document's ids as `uint32` bytes as it goes, renames the temporary file on success, and returns the three counts.
- Full suite in Docker: `PYTHONPATH=src uv run pytest -q` returned `26 passed`.
- Earlier Red iterations during implementation: a missing `:` made `core/data.py` unimportable; `.tobytes()` was applied twice (`AttributeError: 'bytes' object has no attribute 'tobytes'`); the counts were first returned as a set rather than a dict.
- On failure, the temporary file is left behind; only the final path is guaranteed absent. Cleanup is not required by the agreed behavior.

## Manual Real-Data Check — 2026-11-08

The user added the `tokenizers` dependency and ran a one-off script (network and the user's `HF_TOKEN`; not part of the test suite) that connects existing functions to real inputs:

- Tokenizer: `tokenizers.Tokenizer.from_file` on `meta-llama/Llama-3.2-1B` `tokenizer.json` at revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`; `tokenize = lambda t: tok.encode(t, add_special_tokens=False).ids`; `eos_id = tok.token_to_id("<|end_of_text|>")` returned `128001`.
- Dev ids: `dev-ids.txt` from `cerulean-works/fineweb-edu-dev` (revision not yet pinned) contained `5000` ids.
- Data: the first 1,000 rows of `sample/10BT/000_00000.parquet` from `load_dataset`.
- Result: `{'num_docs': 1000, 'num_excluded': 0, 'num_tokens': 1037837}`, about 1,038 tokens per document including EOS. No dev document appeared in these rows, so exclusion was exercised only by the unit tests.
- Round trip: decoding the first document's tokens (up to the first EOS) reproduced the original text exactly (`decode matches original: True`).
- The second `load_dataset` call reported `Fetching 1 files` at about 954 files/s, confirming the cache was reused.

Status: complete for single-process tokenization of real data.

## Wiring (Folded into This Task)

The user chose to turn the one-off script into maintained code within this task and pull request (2026-11-08).

### Agreed Interface

```python
from upwash.core.data import load_dev_ids, prepare_token_shards

dev_ids = load_dev_ids(repo_id, *, revision, filename, cache_dir=None, download=snapshot_download)
# -> set[str]: one id per line, blank lines ignored

summary = prepare_token_shards(
    repo_id, *, revision, data_files, output_dir,
    tokenize, eos_id, exclude_ids, cache_dir=None, download=snapshot_download,
)
# -> {"num_files": ..., "num_docs": ..., "num_excluded": ..., "num_tokens": ...}
```

- One shard per matching parquet file. The shard mirrors the file's path under `output_dir` with a `.bin` suffix (`sample/10BT/000_00000.parquet` → `output_dir/sample/10BT/000_00000.bin`), plus a sidecar `.json` holding that shard's `write_token_shard` counts.
- The `.json` is written after the `.bin` completes; a shard whose `.json` exists is skipped. A rerun after a crash redoes only unfinished files.
- The summary totals all shards' `.json` files, including skipped ones.
- Package functions take every value as a parameter. Real values (repos, revisions, paths) and tokenizer construction live in `scripts/prepare_fineweb.py` as constants for now; config and builder replace them later.
- Local output directory: `.cache/tokens/` (ignored by git).

Deferred: parallel processing, an error when no file matches (beyond `load_dataset`'s own), and pinning the `dev-ids.txt` revision (needed for the script, not the tests).

### Tests

`tests/unit/core/test_prepare_token_shards.py`, with a fake downloader and fake tokenizer:

- `load_dev_ids` returns `{"dev-1", "dev-2", "dev-3"}` from a file with a blank line and requests only the pinned file.
- Two matching parquet files produce `000.bin` `[104, 105, EOS]` and `001.bin` `[121, 111, 33, EOS]`; the `sample/100BT` file produces no shard; every download request uses the pinned revision.
- Sidecar `.json` counts per shard, and summary `{"num_files": 2, "num_docs": 2, "num_excluded": 1, "num_tokens": 7}`.
- A second run tokenizes nothing and returns the same summary.
- After a crash on the second file, a rerun tokenizes only `"yo!"` and returns the full summary.

### Verified Red — 2026-11-08

- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_prepare_token_shards.py`.
- Result: `ImportError: cannot import name 'load_dev_ids' from 'upwash.core.data'`; collection error. Missing API only. The rest of the suite: `26 passed`.
- Test sanity check: a scratch reference implementation outside the repository passed all five tests (`5 passed`).

### Verified Green — 2026-11-08

- The user implemented both functions in `core/data.py`. `load_dev_ids` reads the file with `read_text`, splits lines, and adds non-empty lines to a set. `prepare_token_shards` globs matching files in sorted order; for each one it derives `rel`, `bin_path`, and `json_path`. If the `.json` is missing, it creates the parent directory, reads only that file through `load_dataset`, calls `write_token_shard`, and writes the counts as JSON. It then always reads the `.json` back and adds it to the totals.
- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/unit/core/test_prepare_token_shards.py` returned `5 passed`. Full suite: `31 passed`.
- Red iterations during implementation:
  - `load_dev_ids` returned `None` (missing `return`).
  - `with_suffix("bin")` raised `ValueError: Invalid suffix 'bin'`.
  - Other defects were fixed before tests could reach them: the per-file block sat outside the loop, `cache_dir`/`download` were not forwarded to `load_dataset`, `json.dunps` was misspelled, `num_files` was added in place of `num_docs`, and `return total` was missing.
- Before the implementation, the skip-on-second-run test passed vacuously (both runs returned `None` without tokenizing); it now passes against real shard output.

### Script and Real-Data Run — 2026-11-08

- The user wrote `scripts/prepare_fineweb.py`. It holds the real values as constants:
  - data: `HuggingFaceFW/fineweb-edu` at `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
  - dev ids: `cerulean-works/fineweb-edu-dev` `dev-ids.txt` at `2484afe000a6fdf442ba2396391a3bec12644903`;
  - tokenizer: `meta-llama/Llama-3.2-1B` at `4e20de362430cd3b72f300e6b0f18e50e7166e08`;
  - paths: `.cache/huggingface` and `.cache/tokens`.
- The script builds the tokenizer from `tokenizer.json`, asserts `eos_id == 128001`, then calls `load_dev_ids` and `prepare_token_shards` and prints the summary. The assertion caught a misspelled EOS name (`token_to_id` returned `None`) before tokenization began.
- Command: `uvd run python scripts/prepare_fineweb.py` with `DATA_FILES = "sample/10BT/000_00000.parquet"`.
- Summary: `{'num_files': 1, 'num_docs': 726000, 'num_excluded': 0, 'num_tokens': 731876929}`, about 1,008 tokens per document including EOS. Wall time was about 31 minutes, judging from the output directory and shard timestamps (single process, Docker on an Intel Mac).
- Agent check of the output:
  - `000_00000.bin` is 2,927,507,716 bytes, which is 4 × `num_tokens`.
  - The sidecar `.json` matches the summary.
  - The shard read through `np.memmap` contains exactly 726,000 EOS tokens, which equals `num_docs`.
  - The maximum id is 128,001 and the last token is EOS.
- No dev-set document occurred in this file (`num_excluded: 0`); exclusion on real data remains unobserved.

Next: rerun the script to confirm the completed shard is skipped and the same summary is returned. Then commit and open the pull request.

## Review Changes — 2026-11-08

The teammate's review of the pull request raised four points:

1. **Return type annotations.** Accepted. The user added `-> set[str]` to `load_dev_ids` and `-> dict[str, int]` to `prepare_token_shards` and `write_token_shard`. No behavior change, so no test.
2. **The `.json` skip check is MVP-grade.** Accepted as follow-up work. The `.json` records no source file, revision, tokenizer, or exclusion set, so changing any of them still skips stale shards. A missing or truncated `.bin` is not detected either. Planned: store those inputs in the `.json`, and verify them along with `.bin` size equal to `4 * num_tokens` before skipping.
3. **`download` is called once per file in addition to the snapshot request.** Accepted. The actual call count is `1 + N`, because `prepare_token_shards` calls `load_dataset` for each file. Agreed fix: extract the per-file parquet reading into a shared helper, so `prepare_token_shards` downloads once and no longer calls `load_dataset`. `load_dataset` behavior is unchanged.
   - New test: `test_prepare_token_shards_requests_the_snapshot_once_for_all_files` expects exactly one download call with the pattern.
   - Red (Docker, full suite): `1 failed, 31 passed`. Observed three calls: the pattern, then `sample/10BT/000.parquet` and `sample/10BT/001.parquet`.
4. **Byte order is implicit.** Accepted. Shards are written and read as explicit little-endian `"<u4"` rather than native `np.uint32`.
   - Tests in `test_write_token_shard.py` and `test_prepare_token_shards.py` now read shards with `"<u4"`. On little-endian hardware (x86, Apple Silicon), native and explicit little-endian produce identical bytes, so this change cannot fail there and **no Red is claimed**; it records the format contract.
   - The reader side (`PackedTokenDataset`) is on the task 009 branch.

Green, 2026-11-08:

- The user extracted `read_parquet_rows(path, columns)`. `load_dataset` now delegates to it with `yield from`; `prepare_token_shards` reads each globbed local `path` with it and no longer calls `load_dataset`. `write_token_shard` writes `"<u4"`.
- Full suite (Docker): `32 passed`.
- Intermediate Red: calling `read_parquet_rows(...)` without `yield from` left `load_dataset` with no `yield`, so it was no longer a generator and returned `None`. Three `load_dataset` tests failed with `TypeError: 'NoneType' object is not iterable`.

Next: commit and push to update the pull request, reply to the review, then rebase `feat/packed-token-dataset` and switch `PackedTokenDataset` to `"<u4"`.

## Original Next Action (Done)

Connect real inputs before calling the task complete for real data: add the `tokenizers` dependency (user decision), tokenize with `meta-llama/Llama-3.2-1B` `tokenizer.json` using `add_special_tokens=False`, read `dev-ids.txt` from `cerulean-works/fineweb-edu-dev`, and run one real parquet file through `load_dataset` → `write_token_shard`, checking that decoded tokens match the source text.
