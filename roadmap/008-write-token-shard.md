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

## Remaining Work for Later Tasks

- Turn the one-off wiring into maintained code: reading `dev-ids.txt` at a pinned revision and constructing the tokenizer function.
- Process whole parquet files and many files in parallel, one shard per input file.

## Original Next Action (Done)

Connect real inputs before calling the task complete for real data: add the `tokenizers` dependency (user decision), tokenize with `meta-llama/Llama-3.2-1B` `tokenizer.json` using `add_special_tokens=False`, read `dev-ids.txt` from `cerulean-works/fineweb-edu-dev`, and run one real parquet file through `load_dataset` → `write_token_shard`, checking that decoded tokens match the source text.
