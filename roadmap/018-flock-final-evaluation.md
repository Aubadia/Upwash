# 018 — Final Flock Evaluation from an In-Memory Model

## Status — 2026-11-13

Current direction: the user stopped the Flock evaluation integration because
setup was too cumbersome and requested a self-written evaluator. The direct-API
plan below is historical research, not active work. No Upwash Flock-evaluation
wrapper or tests were created. Do not continue environment setup or delete the
user's Flock checkout. Custom evaluation scope is being clarified under 017.
Flock experiment logging (task 013) was not part of this evaluation cancellation.

The user prioritizes final Flock evaluation ahead of real-data work for task 017
and selected an in-memory `model` plus `tokenizer`, evaluated after training.
An exported model reference/CLI-only entry point is not the selected interface.

Baseline: `./test.sh` → exit 0, `99 passed in 1.96s`. No final-evaluation tests,
production implementation, dependency/configuration changes, installations,
downloads of model weights/datasets, uploads, or scoring runs have been made.

## Selected Approach — Direct Flock APIs

The user questioned the need for a new Upwash wrapper. The proposed
`upwash.evaluation.flock.evaluate_final` API is withdrawn. Use Flock's existing
functions directly in the eventual experiment/evaluation entry point; do not
write missing-Upwash-API tests or production stubs for that withdrawn proposal.

```python
from flock.eval.model import ScoringModel
from flock.eval.run import score_task
from flock.eval.tasks import TASKS

scorer = ScoringModel(
    model, tokenizer, name="run-name@step-12000", batch_tokens=8192,
)
result = score_task(scorer, TASKS["fineweb_edu"])
print(result.metrics)
```

- This is a documentation example, not executed scoring or a new Upwash API.
  The user continues to own experiment/production implementation.
- Fixed scope: `TASKS["fineweb_edu"]` and all tasks under `SUITES["base_easy"]`
  and `SUITES["paloma"]`, scored once per task name, with no document limit.
  Use the installed pinned registry; do not copy a task-name list into Upwash.
- Require an FP32 model that actually supports at least 8192 context. Preserve
  its device/dtype; do not silently cast weights or merely override its context
  metadata. The caller owns any explicit conversion before final scoring.
- Initially accept a causal LM already compatible with `ScoringModel` and its
  supported tokenizer. A generic tensor-returning Upwash module needs a separate
  adapter decision once the actual model interface exists. Attention masks and
  position IDs must retain their semantics; do not discard them to force a call.
- Require a new or empty output directory. Save Flock prediction files, task
  metrics, suite metrics/hashes, and provenance with Flock's result classes.
  Failure propagates and must not leave a successful full-run summary; partial
  predictions may remain. Do not silently skip failed tasks or resume into a
  nonempty output directory.
- Keep metric definitions in Flock. Unscored grouping suites retain `score=None`.
  Do not synthesize a single combined PPL/BPB score for the three benchmarks.
- `name` labels the in-memory model; it is not proof of immutable weights or a
  Hub revision. Any final checkpoint/run identity must be recorded honestly.
- Automatic tracker upload, checkpoint export, and periodic training evaluation
  are outside this first increment.

`score_task` is the direct entry point for an existing model. Full summary
serialization is additional orchestration with official result/provenance
objects, not a reason to require a new reusable Upwash abstraction.

## Verified Flock Source and Documentation

On 2026-11-13 the hosted docs and public repository identify commit
`be6b483eee053efd08409627f6c4786481b20c0b`. This is a new recorded research target,
not a silent replacement of task 017's older documentation revision. No executable
compatibility with this commit has been established.

Read:

- [In-memory evaluation](https://docs.cerulean.works/flock/main/guide/evals/evaluate-checkpoint/)
- [Python API](https://docs.cerulean.works/flock/main/reference/evals/python-api/)
- [Results and provenance](https://docs.cerulean.works/flock/main/reference/evals/results/)
- [Locked-workspace installation](https://docs.cerulean.works/flock/main/guide/installation/)
- [Versioned run source](https://github.com/cerulean-works/flock/blob/be6b483eee053efd08409627f6c4786481b20c0b/flock/eval/run.py)
- [Versioned suite source](https://github.com/cerulean-works/flock/blob/be6b483eee053efd08409627f6c4786481b20c0b/flock/eval/tasks/suite.py)
- [Versioned result source](https://github.com/cerulean-works/flock/blob/be6b483eee053efd08409627f6c4786481b20c0b/flock/eval/results.py)
- [Versioned scorer source](https://github.com/cerulean-works/flock/blob/be6b483eee053efd08409627f6c4786481b20c0b/flock/eval/model.py)

The HTML docs' source links point to `src/flock/eval/...`, which returned 404.
The repository tree at the same commit verifies actual paths `flock/eval/...`;
those files were fetched and inspected successfully.

Key findings:

- `evaluate(EvalConfig(...))` always loads through
  `ScoringModel.from_pretrained`; it does not accept an already loaded model.
  Do not monkeypatch this loader in production to inject one.
- `ScoringModel(model, tokenizer, name=..., batch_tokens=...)` wraps existing
  weights; `score_task(scorer, task, limit=None, out=...)` returns `TaskResult`
  and can write official per-document predictions, but not `results.json`.
- `Suite.tasks` expands tasks; `Suite.suites` orders nested suites after members;
  `by_name` rejects different definitions sharing a name. `Suite.score(metrics)`
  implements Flock's aggregation and can be reused directly.
- Official `evaluate` assembles suite hashes from each suite's definition and
  member hashes. A memory-based adapter needs equivalent result orchestration;
  this is more than dispatching three `score_task` calls.
- `Provenance.collect`, `SuiteResult`, `Results`, and `Results.write(out)` are
  available. Use them instead of inventing a competing JSON schema. The returned
  Flock schema is version 8 at the recorded commit; do not hardcode a version
  number into an Upwash-written imitation.
- Scorer forward calls use `input_ids`, `attention_mask`, `position_ids`,
  `use_cache=False`, and `logits_to_keep`; outputs expose `.logits` with the
  requested number of positions. It also reads model device/dtype/config.

## Dependency and Verification Plan

The current Upwash environment has no installed Flock, `flock-eval`,
`flock-training`, `flock-hub`, `transformers`, or `safetensors`, and no `flock`
executable. This is an environment prerequisite, not a feature-test Red.

The official install uses the full pinned workspace with
`uv sync --all-packages --locked`. Its `flock-eval` member is version 0.1.0 with
`package=false`; do not assume an unrelated PyPI package named `flock` is the
correct evaluator. Installation/environment changes must first be discussed.
Prefer a separate pinned Flock environment for executable integration checks,
keeping its broad workspace dependencies out of Upwash's ordinary training path.

Next steps:

1. Agree on the dependency/test environment before installing anything. Exercise
   real `ScoringModel`, `score_task`, suite aggregation, and `Results.write`
   with tiny CPU models and local dataset fixtures, without downloading the full
   suites. Check known-value NLL/PPL independently and compare result aggregation
   against the pinned official implementation where appropriate.
2. Verify a tiny direct-API integration run, then actual full-suite scoring only
   when a compatible real FP32 model, compute, and dataset access are available.
   Any limited smoke run remains labeled as such, separate from final results.

Superseded next action: pinned Flock environment/direct-API integration is no
longer active. Continue with the user's self-written evaluation scope under 017.
