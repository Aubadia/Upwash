# 033 — Data Pipeline: Decontamination, Packing and Lance Token Datasets

## Behavior

Prepare training data for the data-curation lab: Paloma 9-source macro BPB ≤ 1.25,
scored by token count, all seeds 42. Each source goes through the pipeline once:

```
raw text (a Flock Lance dataset, or an HF parquet file downloaded one at a time)
  -> Paloma decontamination -> tokenize -> pack into seq_len + 1 windows
  -> a token Lance dataset in Flock S3 (versioned)
```

Training then mixes sources by weight from the token datasets, so changing a mixture
never re-tokenizes. Nothing is kept on `/work` beyond the file being processed.
Preprocessing may run on F1 (CPU cluster) and training on nano4; Flock S3 connects
them.

Branch `feat/data-pipeline`, from `main` at `e2efe61` (PR #16 merged), created with
`--no-track`.

## Measurements behind the design (2026-11-20, scratch scripts outside git)

Slurm job 816203 on nano4 compute node 25a-hgpn007 read Flock datasets through
`flock.hub.s3.get_process(read_only=True)`:

| | fineweb-edu v1 | dclm v1 |
|---|---|---|
| sequential scan | 152,776 docs/s, 757 MB/s | 159,000 docs/s, 881 MB/s |
| random `take` x256, serial | 179 docs/s (1.5 s/call) | 160 docs/s |
| random `take` x256, 8 parallel | 260 docs/s | 288 docs/s |
| Llama 3.2 tokenize, all threads | 7.25M tok/s | 5.51M tok/s |
| Llama 3.2 tokenize, one thread | 658k tok/s | 651k tok/s |
| bytes per token / tokens per doc | 5.05 / 977 | 4.41 / 1,208 |

A 1B run needs about 0.4–0.5M tok/s, about 2 MB/s of text. Sequential reads and
tokenization have orders of magnitude of headroom. Random per-document reads do not, so
training must not shuffle raw documents one by one; it reads packed token windows.

Flock facts:
- Datasets are raw-text Lance tables in `s3://flock-datasets/` (e.g. `fineweb-edu`,
  `dclm`, `dolma-v1_7/{reddit,starcoder,wiki,books,pes2o,...}`, `paloma`).
- `paloma` v1 has 154,440 test and 154,943 validation rows over 16 configs. Flock's
  Paloma suite scores 9 of them on the test split (`docs/source/evals/tasks.md` in the
  Flock repo).
- Temporary S3 keys expire after 8 hours. Long jobs use `s3.get_process()` with
  `FLOCK_TOKEN`.

## Token windows on Flock S3 (measured 2026-11-21, scratch script outside git)

Slurm 820790 on nano4 node 25a-hgpn002 (the first try, 820787, wrote fine but failed on a
read-back bug in the scratch script, so cleanup was added to its start and end):

- Path `s3://flock-datasets/lab7/dxnn1u-asciibase64/bench/`. Write, read back and delete
  all work; 12 objects deleted, 0 remaining.
- 40,000 fineweb-edu docs went through the user's `encode_document` and `pack_windows`:
  39,022,040 tokens became 9,526 windows, with 3,543 dropped, at 563k tok/s on one thread.
- Writing to S3 ran at 156 MB in 12.1 s = 12.9 MB/s per process (10 files). That is
  about 3.2M tok/s, above one-thread tokenization. Round trip OK.

| read pattern | windows/s | MB/s |
|---|---|---|
| random, 1 row per `take` | 47 | 0.8 |
| random, 32 rows per `take` | 96 | 1.6 |
| random, 32 rows per `take`, 8 threads | 445 | 7.3 |
| contiguous 64 rows | 2,299 | 37.7 |
| sequential scan | 24,286 | 398 |

A 1B run on 8 GPUs needs about 100 windows/s (12.5 per rank). Fully random sampling (as
`Permute` does today) works with batched `take` and background prefetch threads: about
4.5× headroom on one reader. Block shuffling (contiguous blocks, 23× headroom on one
thread) is the fallback. Decision for step 4: fully random with batched prefetch.

## Paloma decontamination (Appendix C.1, read 2026-11-20)

Rules from the paper (arXiv 2312.10523v2, Section 3 and Appendix C.1):

- Match at the paragraph level, "newline separated spans of text", by exact match.
- Ignore evaluation paragraphs smaller than 13 Unicode-segmented tokens, and
  paragraphs composed only of punctuation, spaces and emoji.
- Remove whole training documents that contain any contaminated paragraph.
- The paper uses a Bloom filter (Dolma's) without giving its parameters.
- It forgoes decontamination of code (Dolma-100-programming-languages and The Stack).
- Reported removal rates are at most 2.753% (The Pile); C4 is 0.010%. Removal rates
  should be checked on new datasets.

Not specified by the paper, decided 2026-11-20:

| Open point | Decision |
|---|---|
| Token segmentation | UAX #29 word boundaries via the `regex` package (WORD mode). A token is any non-whitespace segment, so punctuation counts. `regex` must be declared in `pyproject.toml`; it is currently installed only transitively |
| "Only punctuation, spaces and emoji" | No letter (`L*`) or number (`N*`) character in the paragraph |
| Normalization | None: exact string match of the newline-split paragraph |
| Data structure | A set of 128-bit paragraph hashes instead of a Bloom filter: same matches, no false positives, tens of MB for millions of paragraphs |
| Splits | Both validation and test: the paper does not say, and both would inflate evaluation |
| Code sources | Not decontaminated, as in the paper; a per-source flag in its configuration. Confirm with the TA |
| Module location | `src/upwash/prep/` (preprocessing, separate from training-time `core`) |

## Plan

| step | behavior | where | status |
|---|---|---|---|
| 1 | Decontamination: paragraphs, Unicode token count, eligibility, `ContaminationIndex` | `prep/decontam.py` | Green (2026-11-21) |
| 2 | Encode documents and pack them into `seq_len + 1` windows | `prep/pack.py` | Green (2026-11-21) |
| 3 | Lance token windows: `write_windows`, `LanceWindows` (schema check, `len`, `read(indices)`) | `core/lance_windows.py` | Red |
| 4 | `LanceTokenDataset` and batched fetching in the training `DataLoader` | `core/data.py` | to design |
| 5 | `MixtureDataset`: per-source weights, seed 42, exact per-source counts | `core/data.py` | to design |
| 6 | Config: Lance sources and weights in `DataConfig` (with the run-config task) | `config.py` | to design |

Prerequisites before step 3:
- A write test to our `lab7/<pair>/` path in `flock-datasets`, then delete it.
- Adding `lance` (and `flock` for credentials) to the Upwash environment. This is a
  dependency change and must be agreed first.

## Agreed Interface — Step 1 (2026-11-20)

```python
from upwash.prep.decontam import (ContaminationIndex, is_eligible_eval_paragraph,
                                  split_paragraphs, unicode_token_count)

split_paragraphs(text) -> list[str]          # split on "\n", drop empty spans
unicode_token_count(text) -> int             # non-whitespace UAX #29 segments
is_eligible_eval_paragraph(paragraph) -> bool  # >= 13 tokens and has a letter or number
index = ContaminationIndex.from_texts(eval_texts)  # one pass over an iterable of texts
len(index)                                   # distinct eligible evaluation paragraphs
index.is_contaminated(document) -> bool      # any paragraph matches exactly
```

## Agreed Interface — Step 2 (2026-11-21)

```python
from upwash.prep.pack import encode_document, pack_windows

encode_document(text, tokenize, *, bos_id, eos_id) -> list[int]
# [bos_id] + tokenize(text) + [eos_id]. `tokenize` is str -> list[int] and adds no special
# tokens (the write_token_shard convention).

pack_windows(documents, *, seq_len) -> Iterator[np.ndarray]
# documents: iterable of token lists. A lazy generator; yields uint32 arrays of seq_len + 1.
```

- Format `[BOS, text, EOS]` (user's choice 2026-11-21). It matches lab6's 205B pool and
  Flock's scoring, which prefixes each document with a start token. `main`'s existing
  `write_token_shard` writes `tokenize(text) + [EOS]` only.
- Window k of one call's stream is `stream[k*S : k*S + S + 1]`, the `PackedTokenDataset`
  slicing. Consecutive windows share one token, so each token after the first is a
  target exactly once.
- Option A, chosen over B (rows of S tokens joined at read time): one window is one Lance
  row. A sample is a single-row read, and rows written by different parallel jobs are
  independent. The cost is one duplicated token per window, about 0.024%.
- The leftover between documents is carried inside the generator. The final partial
  window is dropped, and callers count it as `N - (W*S + 1)`. This matches the existing
  local pool, which drops each `.bin` file's tail.

## Agreed Interface — Step 3 (2026-11-21)

```python
from upwash.core.lance_windows import LanceWindows, write_windows

rows = write_windows(uri, windows, *, seq_len, max_rows_per_file=..., storage_options=None)
table = LanceWindows(uri, *, version=None, storage_options=None)
len(table); table.seq_len; table.version
table.read(indices) -> np.ndarray  # uint32, shape (len(indices), seq_len + 1)
```

- Schema: `tokens: fixed_size_list<uint32, seq_len + 1>`, one window per row.
  Provenance (source, revision, tokenizer, decontamination counts) goes in the dataset
  card, not per row.
- `write_windows` consumes a lazy iterable in bounded batches, with `mode="create"` only:
  an existing dataset raises `OSError` and is never overwritten.
  - A window of the wrong length raises `ValueError` (`seq_len`).
  - No windows raises `ValueError`: an empty job output is treated as a pipeline bug.
  - It returns the number of rows written.
- `LanceWindows` checks that the `tokens` fixed-size-list column exists (`ValueError`
  otherwise), derives `seq_len` from the schema, and reports the opened version.
- `read` keeps the requested order and repeats, which Lance `take` does. It checks bounds
  itself (`IndexError`), because Lance raises unclear `OSError` and `OverflowError`
  there. An empty request gives shape `(0, seq_len + 1)`.
- Module: `core/lance_windows.py`. Training reads it, and the writer sits beside the
  reader so the schema is defined once; `prep/` jobs import it.
- Deferred to the parallel-job step: writing per-job fragments and committing them once.
- Dependency: the user added `pylance>=14.0.3` (Lance 14.0.3 in `.venv`).

## Tests

`tests/unit/prep/test_decontam.py` (CPU, pure Python):

- Paragraph splitting: on newlines, with empty spans dropped.
- Unicode token counts under UAX #29:
  - `Hello, world.` is 4, so punctuation counts.
  - `don't stop` is 2, and `pi is 3.14 here` is 4.
  - The 7 Han characters of `我們今天去學校` are 7 tokens, without any spaces.
- Eligibility:
  - A 13-token paragraph is eligible; a 12-token one is not.
  - Long paragraphs of only punctuation, spaces or emoji (13–39 segments) are not.
  - One number makes a long symbol paragraph eligible.
- Matching:
  - A document is contaminated when any paragraph matches exactly.
  - Matches do not count inside a longer paragraph, after lowercasing, with changed
    whitespace, or for a strict prefix.
  - Short and symbol-only evaluation paragraphs never contaminate (`len(index) == 0`).
  - Distinct paragraphs are counted across texts. An empty index contaminates nothing.
  - A one-pass iterator is accepted, since Paloma is streamed.
- The expected counts were checked against `regex` WORD mode before writing the tests.

`tests/unit/prep/test_pack.py` (CPU, fake `tokenize`; S = 4 or 8):

- `encode_document`:
  - Gives `[BOS] + tokens + [EOS]`; an empty text gives `[BOS, EOS]`.
  - Calls `tokenize` once on the exact text.
- Windows:
  - Tokens 0..12 give `[0..4], [4..8], [8..12]`, and windows run across document
    boundaries.
  - A window needs S + 1 tokens, and the final partial window is dropped.
- Outputs are `uint32` arrays of shape `(S + 1,)` that keep id 128001.
- For random document lengths: W = (N - 1) // S. The windows, joined without their
  shared tokens, equal the stream prefix, and the dropped count is in `[0, S]`.
- Lazy: the first window comes from an infinite generator of documents.
- `seq_len` 0 and −4 raise `ValueError`.
- The tests were checked against a scratch reference implementation outside the repo:
  `14 passed`.

`tests/unit/core/test_lance_windows.py` (CPU, local Lance directories under `tmp_path`):

- A round trip of 10 windows: `uint32`, shape `(10, S + 1)`, with `len` and `seq_len`.
- Order and repeats are kept (`[7, 2, 7, 0]`). Indices 10 and −1 raise `IndexError`. An
  empty read has shape `(0, S + 1)`.
- 25 windows across files of 4 rows still read as one table.
- Writing streams straight from `pack_windows` over a one-pass generator and keeps id
  128001.
- A wrong-length window and an empty input raise `ValueError`.
- A second create raises `OSError` and keeps the first dataset.
- A table without `tokens` is rejected, and a pinned `version=1` is reported.
- The tests were checked against a scratch reference implementation outside the repo:
  `12 passed`.

## Progress

- 2026-11-20:
  - Flock datasets explored and the streaming benchmark run (above).
  - Paloma C.1 read; the open points were decided with the user.
- 2026-11-20, step 1, Red: `./test.sh tests/unit/prep/test_decontam.py` fails at collection
  with `ModuleNotFoundError: No module named 'upwash.prep'`, the intended Red.
- 2026-11-21, step 1, Green:
  - The user added `regex` to `pyproject.toml` and `uv.lock` and implemented
    `prep/decontam.py`.
  - Review of the drafts found:
    - `regex.split`'s list treated as one string.
    - `unicodedata.category` not called on the character.
    - An inverted letter-or-number check ("every" instead of "at least one").
    - `from_texts` missing `cls`.
    - `from_texts` hashing whole texts, because the `split_paragraphs` result was
      discarded.
  - All are fixed. `./test.sh tests/unit/prep/test_decontam.py` gives `18 passed`.
  - Full suite through `sbatch scripts/test.sbatch` (new CPU test launcher, as AGENTS.md
    requires): Slurm 820744 on 25a-hgpn008 gave `414 passed in 34.35s`.
- 2026-11-21, step 2, Red:
  - Interface agreed (option A, `[BOS, text, EOS]`).
  - `./test.sh tests/unit/prep/test_pack.py` fails at collection with
    `ModuleNotFoundError: No module named 'upwash.prep.pack'`, the intended Red.
- 2026-11-21, step 2, Green:
  - The user implemented `prep/pack.py`.
  - Review of the draft: the `while` loop sat outside the `for` loop. Packing then waited
    for all input, and the lazy test with an infinite document stream hung until the
    120 s timeout.
  - After indenting it, `./test.sh tests/unit/prep/test_pack.py` gives
    `14 passed in 0.29s`.
  - Full suite via `sbatch scripts/test.sbatch`: Slurm 820764 on 25a-hgpn004 gave
    `428 passed in 33.48s`.
- 2026-11-21, step 3:
  - The write path is `lab7/dxnn1u-asciibase64/`, and the S3 measurements are above.
  - The user added `pylance`.
  - Red: `./test.sh tests/unit/core/test_lance_windows.py` fails at collection with
    `ModuleNotFoundError: No module named 'upwash.core.lance_windows'`, the intended Red.
- Next: the user implements `core/lance_windows.py`.
- Later refactor, after the new pipeline runs end to end:
  - Move the prep-time functions out of `core/data.py` into `prep/`: `load_dataset`,
    `load_dev_ids`, `prepare_token_shards`, `write_token_shard`.
  - Delete the unused `read_parquet_rows` and the obsolete `scripts/prepare_fineweb.py`.
  - Make `write_token_shard` use `encode_document`.
