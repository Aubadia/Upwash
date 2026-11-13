# 021 — Eval Perplexity

## Behavior

Log `eval/perplexity` beside `eval/loss` for language-model runs. The lab requires both metrics, and Trainer already computes and logs `eval/loss` (task 019, training-loop logging). This task does not include the internal fineweb-edu dev data, which is prepared by a script with existing functions, or the Flock `fineweb_edu` scoring after training.

Branch: stacked on `feat/gpu-bf16` (PR #9, task 020), because both change `trainer.py`.

## Agreed Interface — 2026-11-13

```python
trainer = build_trainer(config, model, next_token_loss, optimizer_factory,
                        logger=logger, count_tokens=True, eval_perplexity=True)
trainer.train(train_loader, eval_batches=dev_loader)
# eval event: {"eval/loss": 2.31, "eval/perplexity": 10.07, ...}
```

- `Trainer(..., eval_perplexity=False)` and `build_trainer(..., eval_perplexity=False)`.
- Opt-in, because `exp(loss)` is a perplexity only for a mean natural-log cross-entropy. Generic losses such as MSE must not get it. The setting sits beside `count_tokens` and `pad_token_id`: it describes a language-model run, not a swept hyperparameter, so it is not in `TrainingConfig`.
- When enabled, every event that carries `eval/loss` also carries `"eval/perplexity": math.exp(eval/loss)` as a Python `float`. This covers both scheduled and final evaluation. With the default `False`, events are unchanged.
- Overflow gives `math.inf` and training continues. A diverged sweep run can reach a loss whose `exp` overflows (above about 709.78), and `math.exp` would raise `OverflowError` and end the run. NaN loss gives NaN perplexity.
- `evaluate()` still returns `{"loss", "num_targets"}`. Perplexity is a logged metric only.
- `evaluate()` weights batch losses by target count, so `exp(eval/loss)` is the token-level perplexity (total NLL / total tokens) when `loss_fn` is a mean cross-entropy.

Alternatives considered and not chosen:

- A logger wrapper that derives perplexity from `eval/loss`. Trainer computes every other metric, so this would split metric computation across two places.
- A marker attribute on the loss function. It is implicit: forgetting it silently drops the metric.
- A general `eval_metrics_fn`. Only one derived metric is needed now.
- A `TrainingConfig` field. It is not an experiment hyperparameter.

Known trade-off: Trainer's constructor now has 10 parameters, three of them language-model specific (`count_tokens`, `pad_token_id`, `eval_perplexity`). Grouping them into one options object is a refactor to consider if more appear.

## Tests

`tests/integration/test_eval_perplexity.py`, with real SGD at `lr=0` and 3 training steps, `eval_steps=2`, and an eval-only logger:

- The model is a zero-initialized `nn.Embedding(8, 8)`. Every prediction is uniform over 8 tokens, so `eval/loss = ln 8` and `eval/perplexity = 8`. This is an independent expected value. Both the scheduled (step 2) and final (step 3) events carry it as a `float` equal to `exp(eval/loss)`.
- The default and an explicit `eval_perplexity=False` log `eval/loss` without `eval/perplexity`.
- A loss offset by +1000 overflows `exp`. Both events log `math.inf`, and training completes 3 steps.
- A NaN loss logs NaN perplexity, and training completes.

`tests/unit/test_builder.py`: `build_trainer(..., eval_perplexity=True)` gives perplexity 8 for uniform logits.

## Verified Red — 2026-11-13

Command (Docker, `upwash-venv-pr5`): `PYTHONPATH=src uv run pytest -q tests/integration/test_eval_perplexity.py tests/unit/test_builder.py`.

Result: `5 failed, 4 passed`.

- The four Trainer cases that pass `eval_perplexity` fail with `TypeError: Trainer.__init__() got an unexpected keyword argument 'eval_perplexity'`.
- The builder case fails with `TypeError: build_trainer() got an unexpected keyword argument 'eval_perplexity'`.
- The default-off case already passes, because today's Trainer never logs perplexity.

All failures come from the missing API. No environment failure occurred.

Rest of the suite, ignoring the new file and deselecting the new builder case: `296 passed`.

Test sanity check: a scratch reference implementation outside the repository passed the full suite (`302 passed`). `_evaluation_metrics` added `math.exp(loss)`, with `OverflowError` mapped to `math.inf`, and the builder passed the flag through.

## Verified Green — 2026-11-14

The user implemented the following:

- `Trainer(..., eval_perplexity=False)` stores the flag.
- `_evaluation_metrics` keeps `evaluate()` inside `interval.exclude_time()`, then adds `math.exp(loss)` when enabled, with `except OverflowError` giving `math.inf`.
- `build_trainer(..., eval_perplexity=False)` passes the flag through.

Command (Docker, `upwash-venv-pr5`): `uv run pytest -q` returned `302 passed`.

Red iterations during implementation:

- First version, `5 errors` at collection:
  - A missing comma after `precision=config.precision` in `build_trainer` was a `SyntaxError`.
  - `_evaluation_metrics` also had four defects:
    - It stored the whole `evaluate()` dict as `eval/loss`.
    - It compared the flag with the string `"True"`.
    - It used a bare `except` and `math.inf(loss)` (not callable).
    - It re-raised `OverflowError`.
- Second version, `25 failed, 277 passed`. The Trainer changes were correct, and all five perplexity integration cases passed.
  - `eval_perplexity=False` had been added to the `optimizer_factory` call: `TypeError: SGD.__init__() got an unexpected keyword argument 'eval_perplexity'` in 24 tests.
  - `build_trainer` didn't accept the parameter, which caused the builder case's `TypeError`.

Cosmetic, open:

- `eval_perplexity = False` in the `build_trainer` signature has spaces around `=`, and a blank line remains before `) -> Trainer:`.
- `if self.eval_perplexity == True:` can be `if self.eval_perplexity:`.
- `eval_perplexity=False` in `Trainer.__init__` has no trailing comma.

## Internal Dev Data — 2026-11-14

The user wrote `scripts/prepare_dev.py`. It calls the existing `prepare_token_shards` on `cerulean-works/fineweb-edu-dev` at revision `2484afe000a6fdf442ba2396391a3bec12644903`, file `data/validation-*.parquet` (columns `id`, `text`, `url`, `dump`). Other settings:

- Tokenizer `meta-llama/Llama-3.2-1B` at `4e20de362430cd3b72f300e6b0f18e50e7166e08`, with BOS prepended in `tokenize`. `write_token_shard` appends EOS, so each document is `[BOS, text, EOS]`, like the training data.
- `exclude_ids=set()`.
- Output in `.cache/dev-tokens/`, separate from the training shards in `.cache/tokens/`, so a training glob can't pick up dev data.

The user's Mac is Intel (`macosx_15_0_x86_64`), and torch 2.15.0 has no wheel for it. The script therefore ran in Docker (`upwash-venv-pr5`). A native `uv run` attempt removed the local `.venv` and failed to recreate it. Use Docker for tests and scripts on this machine.

The user ran it and got `{'num_files': 1, 'num_docs': 5000, 'num_excluded': 0, 'num_tokens': 5360078}`.

Agent check of `.cache/dev-tokens/data/validation-00000-of-00001.bin`, read-only:

- 5,360,078 tokens, with 5,000 BOS (128000) and 5,000 EOS (128001). The file starts with BOS and ends with EOS. Every BOS after the first follows an EOS, and every EOS before the last is followed by a BOS. The maximum id is 128001.
- Document length in tokens: minimum 65, median 732, maximum 102,722. 22 documents are longer than 8192.
- `PackedTokenDataset(path, 4096)` gives 1,308 samples, and the final 2,509 tokens are dropped as a partial sample. A `DataLoader(batch_size=8)` batch is `(8, 4096)` `int64` and starts with BOS, with `y` shifted by one.

Rough cost: one evaluation is a forward pass over 5.36M tokens, about `2·N·D` = 1.6e15 FLOPs at 150M and 1.1e16 at 1B. That is seconds to tens of seconds on one H100, so the full set can be used at every evaluation.

Status: code and tests complete, and dev data prepared. On branch `feat/eval-perplexity`, not yet committed, including `scripts/prepare_dev.py`. Next for internal evaluation: wire `dev_loader` and `eval_perplexity=True` into the entry script, with `eval_steps` at most 10% of total steps. Next for Flock: `fineweb_edu` scoring after training.
