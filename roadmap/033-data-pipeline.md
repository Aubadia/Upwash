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
| 3 | Lance token windows: `write_windows`, `LanceWindows` (schema check, `len`, `read(indices)`) | `core/lance_windows.py` | Green (2026-11-22) |
| 4 | Per-fragment processing job (decontaminate, encode, pack, write fragments) and one commit job; option B | `experiments/` or `scripts/` (vibe-coded on agreement) plus a fragment writer in `core/lance_windows.py` | to design |
| 5 | Training reads Lance: `LanceWindowDataset`, `Permute`, `DataConfig(format="lance", version, shuffle_seed)` | `core/`, `config.py`, `builder.py` | Green (2026-11-22) |
| 6 | `MixtureDataset` and config for several sources (deferred until a second source) | `core/data.py`, `config.py` | deferred |

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

## Decisions — 2026-11-22

- Baseline source: DCLM on Flock (`dclm` v1: 2,949,254,346 docs; about 1,208 tokens/doc and
  3.5T tokens in total). Paloma decontamination applies, since it is web text, not code.
- Code sources (e.g. starcoder) are not decontaminated, per Paloma C.1. Open risk: Paloma's
  `dolma_100_programing_languages` documents may overlap The Stack, so that source's BPB
  could read optimistic. Ask the TA.
- Step 4 output layout, option B: each job writes fragments only, and one commit job
  creates a single dataset version. This avoids opening thousands of per-job datasets at
  training start (about 0.7 s each on S3). The Lance APIs (`lance.fragment.write_fragments`,
  `LanceDataset.commit`) still need a scratch check on Lance 14.0.3.
- Step 4 may be vibe-coded as experiment tooling if the host agrees. It only calls the
  hand-written steps 1–3, outside `src/`, with tests for token accounting, applied
  decontamination and seeded fragment selection. Step 5 is hand-written (training path).
- Step 5 reuses `DataLoader`, `RankShard`, `Truncate`, `ConcatDataset` and PR #17's
  `CUDAPrefetchLoader`, whose worker thread already reads the next batch in the background.
  S3 credentials come from `flock s3 configure` plus `AWS_PROFILE`, so there is no `flock`
  dependency in Upwash. Batched reads (one `take` per batch) are deferred until an 8-GPU
  measurement. The estimate: per-sample reads (about 21 ms each) fit a 1B step (about
  330 ms) but not a 150M step (about 48 ms).

## Agreed Interface — Step 5 (2026-11-22)

```python
from upwash.core.data import Permute
from upwash.core.lance_windows import LanceWindowDataset

Permute(dataset, seed)                 # one fixed permutation of the whole pool, the same on every rank
LanceWindowDataset(uri, *, version=None)
len(dataset); dataset.seq_len
dataset[i] -> (inputs, targets)        # int64 tensors: row[:-1], row[1:]; negative indices allowed

DataConfig(root, pattern, format, seq_len, num_tokens, shuffle_seed=None, version=None)
# format "lance": root is the dataset URI; pattern is unused (""). version only for "lance".
# build_dataset: pool -> Permute(shuffle_seed) if set -> Truncate(num_tokens // seq_len)
```

- Shuffle before truncation, so a shorter budget reads a prefix of the same permutation,
  as on `experiment`. With `shuffle_seed=None`, storage order is unchanged, so existing
  configs keep their behavior.
- `build_dataset` rejects a Lance dataset whose `seq_len` differs from the config.
- A `version` with `npy`/`bin` raises `ValueError`, because local files have no versions.
- `to_dict()` now includes `shuffle_seed` and `version`. Its existing test was updated
  for the changed interface.

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

Step 5 tests (CPU, local Lance directories):

- `tests/unit/core/test_permute.py`:
  - A seeded permutation of every sample.
  - The same seed gives the same order, and another seed gives a different one.
  - The length is kept, and an index past the end raises `IndexError`.
- `tests/unit/core/test_lance_window_dataset.py`:
  - A sample is the window shifted by one token, as `int64`, with negative and
    out-of-range indices handled.
  - `DataLoader` batches `(2,S),(2,S),(1,S)`.
  - `build_dataset(format="lance")`:
    - applies the token budget and rejects a budget larger than the data;
    - rejects a `seq_len` mismatch;
    - pins `version=1` after an append;
    - gives a shuffled prefix property and is deterministic.
  - Two `RankShard`s partition it.
- `tests/unit/test_data_config.py`:
  - `"lance"` is accepted, and both new fields default to `None`.
  - A Lance config may pin a version and a seed. A version on `npy`/`bin` is rejected.
  - `to_dict` includes the new fields.
- `tests/unit/test_build_dataset.py`:
  - A shuffled `npy` budget is a prefix of the same permutation; another seed differs.
  - No seed keeps storage order.
- The tests were checked against a scratch reference copy of `src/` outside the repo:
  `47 passed in 8.61s`.

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
- 2026-11-22: merged `main` at `6142f34` (PR #17) and renumbered this task 028 → 033.
- 2026-11-22, step 3, Green:
  - The user implemented `core/lance_windows.py`.
  - Review fixes:
    - The first batch is now validated before Lance starts writing, because Lance wraps
      generator exceptions as `OSError`.
    - The row count comes from the returned dataset.
    - Fixed: an off-by-one bounds check, a column-name typo, and a loop that never
      fetched the next batch.
  - `./test.sh tests/unit/core/test_lance_windows.py` gives `12 passed`.
  - Full suite via `sbatch scripts/test.sbatch`: Slurm 839332 gave `458 passed in 66.73s`.
  - Limit: a wrong-length window in a later batch surfaces as Lance's `OSError`, not
    `ValueError`. Lance commits nothing in that case.
- 2026-11-22, step 5, Red:
  - `test_permute.py` fails at collection: `cannot import name 'Permute'`.
  - `test_lance_window_dataset.py` fails at collection: `cannot import name
    'LanceWindowDataset'`.
  - `test_data_config.py` gives `6 failed, 19 passed`: no `shuffle_seed`, `version` or
    `"lance"` yet.
  - `test_build_dataset.py` gives `1 failed, 9 passed`: no shuffling yet.
- 2026-11-22, step 5, Green:
  - The user implemented `Permute` (`core/data.py`), `LanceWindowDataset`
    (`core/lance_windows.py`), the `DataConfig` fields and checks, and the lance branch of
    `build_dataset`.
  - Review fixes: `Permute` stored no `dataset` and returned the index instead of the
    sample; an attribute-name mismatch in `LanceWindowDataset`; a `config.format` typo.
  - The four step 5 test files give `47 passed`.
  - Full suite via `sbatch scripts/test.sbatch -q`: Slurm 839703 gives 477 passed, with no
    failures, errors or skips. The doubled `-q` suppresses pytest's summary line, so the
    count is from the progress dots.
- Agreed split (2026-11-22): steps 1–3 and 5 (hand-written) go to one PR. Step 4
  (vibe-coded if the host agrees) goes to a separate branch from `feat/data-pipeline`.
- Next:
  - Open the PR for steps 1–3 and 5.
  - Step 4 is designed after the host confirms whether it may be vibe-coded. First check
    Lance 14.0.3 `lance.fragment.write_fragments` and `LanceDataset.commit` in a scratch
    script.
  - After step 4: an 8-GPU 150M smoke reading Lance from Flock S3, to decide whether
    batched reads are needed.
- Later refactor, after the new pipeline runs end to end:
  - Move the prep-time functions out of `core/data.py` into `prep/`: `load_dataset`,
    `load_dev_ids`, `prepare_token_shards`, `write_token_shard`.
  - Delete the unused `read_parquet_rows` and the obsolete `scripts/prepare_fineweb.py`.
  - Make `write_token_shard` use `encode_document`.


## DCLM 150M raw streaming baseline — 2026-11-23

- The experiment branch adds `scripts/train_dclm_150m.py` and
  `scripts/dclm_150m.sbatch`. It reads raw DCLM v1 with Flock's renewable
  credential process, excludes Paloma validation/test paragraph matches, and
  streams deterministic packed batches to eight ranks. This experiment uses
  branch-output residual scaling of `1/6` for both attention and MLP.
- The horizon is 228,881 updates and 29,999,890,432 non-padding tokens.
  Checkpoints include the completed step and next data cursor. All run
  artifacts are under `/work/asciibase64/upwash/dclm-150m/`.
- `sbatch --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 scripts/dclm_150m.sbatch --test`
  (845340) passed the initial four tests. The same resource overrides with
  `--preflight` (845341) read a text batch and tokenized one row from each
  pinned Lance snapshot. The Paloma split column is `_hf_split`.
- `sbatch --gpus-per-node=1 --cpus-per-task=8 --time=00:15:00 scripts/dclm_150m.sbatch --test tests/unit/prep tests/unit/components/schedulers tests/unit/test_adaptor.py tests/integration/test_ddp_training.py`
  (845350) passed 58 tests.
- The first eight-GPU smoke (845345) was cancelled after one failed rank
  blocked in NCCL communicator teardown and hid the original exception.
  Exception handling now logs the worker error before torchrun cleans up.
  Its dependent resume (845346) was also cancelled.
- `sbatch --time=00:20:00 scripts/dclm_150m.sbatch --run-id smoke-2 --stop-after 200`
  (845364) completed 200 updates and 26,214,400 tokens. The final 100 updates
  ran at 1.440M tokens/s, with training loss 8.5868 and Paloma validation
  subset BPB 3.6417. At that rate the training updates alone project to
  about 5 hours 47 minutes.
- `sbatch --dependency=afterok:845364 --kill-on-invalid-dep=yes --time=00:15:00 scripts/dclm_150m.sbatch --run-id smoke-2 --resume --stop-after 201`
  (845365) matched the checkpoint's next-batch hash and three peak LRs, then
  completed step 201 with 26,345,472 cumulative tokens and validation subset
  BPB 3.6410. The smoke checkpoint is `runs/smoke-2/latest.pt` under the
  experiment's work directory.
- The full run is on hold. The user asked to reuse the existing public
  `upwash.train.train` entry point. The experiment script already uses
  `Trainer.train` for the update loop, but duplicates the setup and
  run lifecycle in the public entry point. Next, align the extension points
  for streaming batches, the optimizer recipe, and checkpoint/resume.


### Public training entry point integration

- The DCLM script now supplies a per-rank stream, recipe factories, and
  checkpoint and evaluation hooks to `upwash.train.train`. The public entry
  point builds the runtime, model, DDP, optimizer and scheduler and calls
  `Trainer.train` once. WSD checkpoints save the next update index.
- `sbatch --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 scripts/dclm_150m.sbatch --test tests/unit/test_train.py -k public_train_uses_stream`
  (845381) produced the expected Red because `TrainingProgress` did not exist.
- `sbatch --gpus-per-node=1 --cpus-per-task=8 --time=00:15:00 scripts/dclm_150m.sbatch --test tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/components/schedulers tests/integration/test_training_logging.py tests/integration/test_scheduled_evaluation.py tests/unit/prep`
  (845388) passed 103 tests.
- `sbatch --gpus-per-node=1 --cpus-per-task=12 --time=00:20:00 scripts/dclm_150m.sbatch --test tests/`
  (845394) passed 484 tests.
- `sbatch --time=00:20:00 scripts/dclm_150m.sbatch --run-id upwash-entry-smoke --stop-after 500`
  (845391) completed 65,536,000 tokens. With bounded 16-batch prefetch,
  the post-warmup intervals ran at 1.846–2.000M tokens/s. Rank 0 waited
  0.202 seconds for prefetched data across the entire smoke. The slowest
  steady interval projects to about 4 hours 31 minutes of updates.
- `sbatch --dependency=afterok:845391 --kill-on-invalid-dep=yes --time=00:15:00 scripts/dclm_150m.sbatch --run-id upwash-entry-smoke --resume --stop-after 501`
  (845393) matched the saved next-batch hash, restored all three LRs, and
  completed the next update for a total of 65,667,072 tokens.
- `sbatch scripts/dclm_150m.sbatch --run-id dclm-v1-150m-seed42` submitted the
  eight-H200, five-hour full run as 845412. Output directory is
  `/work/asciibase64/upwash/dclm-150m/runs/dclm-v1-150m-seed42/`.
- Slurm 845412 is running. Its first periodic checkpoint was written at
  step 5,000 to `latest.pt` (1,592,739,591 bytes). At that update it had
  processed 655,360,000 tokens, with training loss 4.1372 and 1.945M
  tokens/s over the preceding logging interval. The prescribed endpoint
  and `final.pt` remain pending.

### Flock logging and Lab 7 OJ fields

- The submitted training job writes `metrics.jsonl`. A separate lightweight
  `scripts/sync_dclm_flock.py` process backfilled it into
  [Flock run d845412](https://flock.cerulean.works/cerulean-labs/lab7-data-curation/runs/d845412)
  and follows new records under the same run ID. It runs in the
  `upwash-flock-845412` tmux session without interrupting training.
- The Flock HTTP API read-back verified 137 training-loss points from
  step 100 through 13,700 and the corresponding 1,795,686,400 cumulative
  tokens in the run summary. Synchronizer output is in
  `/work/asciibase64/upwash/dclm-150m/logs/flock-845412.out`.
- The Lab 7 OJ reads `oj_total_tokens`, `total_tokens`, and `paloma_ppl`
  from the Flock run summary. The synchronizer sets `total_tokens` after
  Slurm reports completion and the final checkpoint and metric step exist.
  `eval/bpb` is the 128-document validation subset used for training
  diagnostics. It is separate from the final qualifying Paloma score.
- The first Paloma smoke (845487) restored the checkpoint but could not fetch
  the gated Hugging Face dataset. The retry (845491) staged Flock's copy of
  `allenai/paloma@65cd6fc59dba021b21db414fa5e8d7765ffbe5e6` and passed
  one pinned test document per task, covering 309 tasks and nine sources.
- `sbatch --dependency=afterok:845412 --kill-on-invalid-dep=yes scripts/eval_dclm_paloma.sbatch <run-dir>/final.pt --out <run-dir>/paloma-final --flock-run-id d845412`
  queued the full evaluator as 845506. It uses the saved model with Upwash's
  residual and logits scaling and writes Flock task and source results under
  the run directory. It publishes the mean of the nine source BPB scores as
  `paloma_ppl` per the Lab 7 macro BPB convention. For this single-pass
  DCLM stream, both token fields use the verified count in the final
  checkpoint. It reads the three summary values back from Flock after upload.
  The full evaluation and summary read-back remain pending.


### Spectral Muon and width-scaled decay

- New DCLM runs pass one Muon group at base LR `0.0017` with
  `adjust_lr_fn="spectral_unclamped"` and momentum `0.9`. AdamW retains
  `0.00272` and betas `(0.9, 0.99)`. Log the group LRs before Muon's
  internal shape adjustment.
- Following [Qiu et al., §3.4](https://proceedings.nips.cc/paper_files/paper/2025/file/bdcd9f6327db5877dee502cdec183159-Paper-Conference.pdf)
  and the [reference scheduler implementation](https://github.com/charliezchen/scaling-matrix-preconditioning/blob/main/train.py),
  apply independent decay `s(t) * c / d_model` before the parameter update,
  with the same WSD factor `s(t)` for the gradient update. At width 768,
  `c=0.0175` gives Muon peak decay `2.278645833e-5`. Treating the AdamW
  surface value `2.63e-5` as its own `c` gives peak decay `3.424479167e-8`.
- The in-flight baseline (845412) retains its recorded optimizer recipe.
  Resuming a checkpoint with its original group layout uses that recipe.
  Fresh runs now use spectral Muon.
- `sbatch --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 scripts/dclm_150m.sbatch --test -k optimizer_partitions`
  (845590) reproduced the intended Red because the old groups used
  `original` instead of `spectral_unclamped`. The first regression run
  (845593) had 39 passes and one reference comparison failure caused by
  PyTorch's default `weight_decay=0.1` in the test oracle. After setting the
  oracle's built-in decay to zero, the regression rerun (845596) passed all
  40 tests.
- The first eight-H200 smoke (845594) encountered a Flock S3 manifest read
  error before training. The retry (845601) on a different node completed
  200 steps and 26,214,400 tokens. The final metrics logged Muon peak LR
  `0.0017`, AdamW peak LR `0.00272`, and finite training loss `8.5488`.
  Command: `sbatch --exclude=25a-hgpn173 --time=00:15:00 scripts/dclm_150m.sbatch --run-id spectral-unclamped-smoke-retry --stop-after 200`.
  Next, use this recipe for fresh model runs and retain the original recipe
  for resuming 845412.


### Fresh spectral baseline with validation at each checkpoint

- Cancelled the original training (845412) and its dependent evaluation
  (845506) at the user's request. The original run logged through step
  52,000, and its most recent saved checkpoint is from step 50,000. Flock
  closed `d845412` and retained its history.
- A single callback saves and evaluates at steps 5,000, 10,000, and all
  subsequent checkpoint intervals, plus the terminal step. It evaluates
  the fixed 128-document Paloma validation subset and logs the result on
  the checkpoint's training step. The endpoint alone runs the full
  309-task Paloma test suite in a separate Slurm job.
- `sbatch --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 scripts/dclm_150m.sbatch --test -k validation_runs_once_per_saved_checkpoint`
  (845615) reproduced the intended Red. Step 5,000 saved a checkpoint
  without evaluating it. The checkpoint, trainer, and WSD regression
  `sbatch --gpus-per-node=1 --cpus-per-task=6 --time=00:10:00 scripts/dclm_150m.sbatch --test tests/unit/components/schedulers tests/unit/test_train.py`
  (845621) passed all 41 tests.
- `sbatch --dependency=afterok:845621 --kill-on-invalid-dep=yes --exclude=25a-hgpn173 scripts/dclm_150m.sbatch --run-id dclm-v1-150m-spectral-seed42`
  started fresh training as 845622. Its dependent endpoint evaluation is
  845623. Flock synchronization is running in `upwash-flock-845622`, using
  run ID `d845622`. The new output directory is
  `/work/asciibase64/upwash/dclm-150m/runs/dclm-v1-150m-spectral-seed42/`.
- The first fresh checkpoint (step 5,000) was saved after 655,360,000
  tokens and evaluated once. The Flock history API returned one point at
  that step for validation loss `3.405684790` and BPB `1.465683402`.
  Training is still running, and the full endpoint evaluation is pending.


### Standalone checkpoint validation — 2026-11-24

- The spectral baseline (845622) finished all 228,881 steps and
  29,999,890,432 tokens. The separate official Paloma job (845623) completed
  with nine-source macro BPB `1.123023455` and verified the OJ summary in
  [Flock run d845622](https://flock.cerulean.works/cerulean-labs/lab7-data-curation/runs/d845622).
- `sbatch --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 scripts/dclm_150m.sbatch --eval-checkpoint <checkpoint.pt>`
  loads checkpoint weights, reconstructs residual and logits scaling, and
  reuses the in-training Paloma validation scorer. Flock supplies the pinned
  validation data; the checkpoint replay invokes no Flock evaluator.
- One-H200 replay of the spectral baseline's `final.pt` (850393) matched
  all of its in-training validation fields exactly. Loss `2.801663344`,
  BPB `1.205734445`, 218,850 tokens and 733,644 bytes.


### DataDecide 1B recipe replay — 2026-11-24

- Selected workbook 1B MacroRank 1–10 and the three released seeds at
  step 69,369. The manifest records the model commit for each of the
  30 checkpoints and the ordered Paloma v1 validation snapshot.
  It contains the same 128 documents and 733,644 bytes as the baseline.
- The scorer reuses Upwash validation with Ai2's native OLMo loader,
  EOS prefix, 2,048-token context and BF16 autocast. The isolated workspace
  environment uses Python 3.11.18, PyTorch 2.6.0+cu124, Transformers 4.50.3,
  ai2-olmo 0.6.0, NumPy 1.26.4, datasets 3.5.0, PyArrow 19.0.1 and pandas
  2.2.3. The loader imports datasets at runtime. Its NumPy pin requires
  compatible Arrow and pandas revisions.
- `bash scripts/submit_datadecide_paloma.sh <workbook> datadecide-1b-top10-20261124`
  submitted ten eval jobs (850550–850558 and 850567) and dependent ranking
  job 850569. The partition accepts nine active submissions and runs four
  at a time. The launcher resumed the partial submission after the
  first jobs freed slots, keeping the existing IDs.
- Outputs are in
  `/work/asciibase64/upwash/datadecide-paloma/runs/datadecide-1b-top10-20261124/`.
  All ten evaluation jobs completed and produced 30 seed results.
  Slurm 850569 passed both numerical and ranking tests, then wrote
  `ranking.csv` and `ranking.md`. Independent read-back checked model
  revisions, the common snapshot hash, all means and sample deviations.
  Falcon+CC (QC Orig 10%) ranked first with mean BPB 1.030950662. The
  report is a ranking of the selected recipes on this validation subset.


### DataDecide 150M training with confirmed Paloma early stopping

- Reuse the 150M spectral baseline through Upwash's training entry point.
  The recipes are Paloma replay ranks 1, 2, 3, 5 and 6. Rank 4 is the
  already-trained DCLM baseline.
- `sbatch scripts/dclm_150m.sbatch --prepare-datadecide <manifest>` indexes
  the exact files in the pinned Ai2 registry against the Flock DataDecide
  v1 Lance table. The initial resource request was rejected because it
  requested more than 12 CPUs per GPU. The corrected one-GPU, 12-CPU job
  850792 passed and wrote the manifest under
  `/work/asciibase64/upwash/dclm-150m/datadecide-top5/`.
- Reconstruct OLMo documents across Lance row boundaries, apply the existing
  Paloma decontamination, and encode with the baseline Llama 3.2 tokenizer.
  Each run shuffles its files with a fixed seed, scans the selected source
  positions once, and can restore the mixture from its checkpoint.
- Every 5,000 steps, save the checkpoint and run the fixed Upwash Paloma
  validation. At local BPB at or below 1.25, pause all ranks and run the
  full official nine-source Flock scorer on the saved checkpoint. End
  training at that step if official macro BPB is also at or below 1.25.
  Confirm the full suite at the token cap as well. The final checkpoint
  has a second filesystem name for the same bytes, and the Flock logger
  validates the terminal step, token count and matching scores before
  publishing the OJ fields.
- The Red job 850782 failed during collection because the recipe module
  did not yet exist. The Green job 850797 passed 31 targeted tests. Job
  850805 passed 14 tests, including rejection of mismatched Flock
  summaries and the check that the early stop consumes no further
  training batch. Both used the `--test` mode of
  `scripts/dclm_150m.sbatch`.

- Eight-H200 job 850802 completed 50 real updates, saved the checkpoint
  and evaluated the same 128 validation documents. Job 850811 restored
  the data cursor and scheduler, reproduced next-batch SHA-256
  `da409a818a31789223949f31dcb1513a23de217f5b7c34d3a77d356a8891611d`,
  and completed step 51.
- `bash scripts/submit_datadecide_training.sh datadecide-gate-20261124`
  submitted the five production recipes with eight H200s and a six-hour
  allocation each. The job IDs in rank order are 850822, 850823, 850824,
  850825 and 850830. The original rank-6 submission 850826 passed
  `sbatch --test-only` but could not dispatch because `32gpus` requires
  at least 17 GPUs. The account is restricted from other eligible
  eight-GPU queues. The replacement retains eight GPUs and queues in
  `8gpus`, which permits four concurrent jobs per user. All five
  production jobs are independent.
- The batch ledger is
  `/work/asciibase64/upwash/dclm-150m/datadecide-top5/datadecide-gate-20261124.tsv`.
  Flock synchronization runs in one tmux session per Slurm job. API
  read-back confirmed the first four training runs and the first three
  step-5,000 validation BPBs at 1.518925795, 1.543089046 and 1.515096853.
  Each has its corresponding checkpoint. The rank-6 job is queued.
- Full-score confirmation smoke 850817 completed the nine-source
  evaluator on the existing 150M baseline, verified the existing
  1.123023455 macro BPB and its qualification, and reused the result
  only after validating the checkpoint digest and progress. Production
  Paloma results and the final OJ summaries remain pending.


### Full-horizon DataDecide training

- Train each recipe to 228,881 steps (29,999,890,432 tokens) by default.
  The fixed validation subset runs at each 5,000-step checkpoint.
  The full nine-source Flock Paloma suite scores the final checkpoint.
  The previous BPB-gated behavior is available through `--early-stop`.
- The Paloma execution policy stays outside the checkpoint configuration.
  Resume restores the original schedule, optimizer, RNG and data cursor.
  The Flock sidecar can attach a replacement Slurm ID to the original
  Flock run and waits for recovery to reconcile the metric file before
  uploading new steps.
- Slurm test command
  `sbatch --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 scripts/dclm_150m.sbatch --test tests/unit/test_train_datadecide_recipes.py`.
  Job 851338 passed all 18 tests. The queued rank-6 job 850830 started
  with the new default through the existing entry point.
- The four earlier runs handed off at a recorded checkpoint. Ranks 1, 2
  and 3 resume from step 100,000 under jobs 851389, 851350 and 851357.
  Rank 5 resumes from step 95,000 under 851341. Each replacement retains
  the original Flock ID. The 8gpus concurrency limit queued 851389.
- The checkpoint cutover and endpoint monitor record replacement job IDs
  and verify each full checkpoint against the matching official result
  and Flock OJ metrics under
  `/work/asciibase64/upwash/dclm-150m/datadecide-top5/full-horizon-20261124/`.
- Flock read-back confirmed continued training after the three active
  checkpoint resumes and confirmed metrics for rank 6. The first
  post-cutover read-back saw d850823 at 100,700, d850824 at 100,300, d850825 at 95,900, d850830 at 4,400.
  Final Paloma and OJ results are pending.
