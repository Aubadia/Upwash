# 017 — Small Development Evaluation

## Status — 2026-11-13

The user accepted the small local evaluation plan after the Flock documentation
review. The first increment is manual `Trainer.evaluate(batches)`, returning
`{"loss": float, "num_targets": int}` with target-weighted mean loss; LM callers
derive perplexity. Final full scoring remains with Flock. The user has added
`evaluate`; the manual evaluation behavior is now Green for its tested scope:
weighted aggregation/types, eval mode/no-grad, normal/loss-error mode restoration,
empty-input rejection, training-state preservation, and tiny LM integration.
Real dev subset preparation/evaluation and automatic scheduling remain pending.
The agent has changed only tests/documentation, not production code.

Session baseline on branch `feat/Trainer-Log`, HEAD `1d2dc74`:
`./test.sh` → exit 0, `92 passed in 1.99s`. This verifies existing behavior only,
not evaluation. At that baseline Trainer exposed `compute_loss` and `train`, and
stored an optional logger without emitting metrics. Latest evidence is below.

## Agreed Initial Scope

1. Add a manually invoked `Trainer.evaluate(batches)` in core before automatic
   scheduling. Reuse the supplied model/loss and existing `(inputs, targets)`
   convention. Leave optimizer construction and concrete loggers independent.
2. Initially support dense, unpadded targets with a scalar loss that is the mean
   over target elements. This is an additional evaluation contract: the current
   scalar-loss contract does not specify reduction. Weight each batch loss by
   `targets.numel()`; divide the total weighted loss by total target count.
3. Result: `{"loss": float, "num_targets": int}`. The LM caller derives
   perplexity as `exp(loss)` only when loss is natural-log next-token
   cross-entropy. Arbitrary supplied losses (including existing MSE/L1 examples)
   must not automatically become a perplexity metric.
4. Use evaluation mode and disable gradient recording. Preserve model parameters,
   existing gradients, optimizer state, and training progress. Restore the prior
   model mode after success or failure. Initially assume uniform module mode;
   deliberately mixed per-module modes need a separate decision.
5. Reject zero evaluated targets with `ValueError`. No padding/ignored-label
   counting, variable-length collation, distributed reduction, BPB calculation,
   automatic triggers, or Flock invocation in this first increment.

Packing already shifts targets in `PackedTokenDataset`; evaluation must not shift
them again. The tiny LM test's `mean_loss` helper averages batch means; its current
fixture has equally sized batches, but that approach is unsuitable for a general
evaluation loop with a smaller final batch.

## Development Data Proposal

Prepare a fixed subset of the designated FineWeb-Edu development set, cache its
tokens, and reuse the existing packed dataset/loader. Start with a tunable budget
such as 128 sequences of length 4096 (524,288 scored target tokens). Keep document
selection, dataset revision, tokenizer revision, BOS/EOS policy, sequence length,
packing, and evaluated target count fixed across comparisons. Select documents
once, rather than resampling at each evaluation. Match the chosen training data's
tokenizer and special-token policy. These local numbers are development metrics;
packing and context rules differ from Flock's final scoring, as confirmed below.
This packed-data proposal remains a cheap training diagnostic only. A closer
local approximation would preserve document boundaries and the official prefix
and target rules while reducing the document count. That requires a separate
data/windowing decision; do not silently add padding to the unpadded API above.

All 5,000 official dev documents must remain excluded from training, even when
local evaluation uses only a subset. Our own preprocessing accepts dev IDs; the
exclusion status of Flock's pre-tokenized shards remains unverified (task 016).

## Proposed TDD Sequence

Begin with the smallest numerical case: known losses with
unequal batch sizes produce the target-weighted result, returned as detached
Python values. Then cover evaluation mode/no gradient recording, preservation of
existing gradients/parameters/progress, mode restoration including errors, empty
input, and a tiny real LM integration. Use independently calculated examples;
this task does not claim Hugging Face Trainer or Flock compatibility.

## Current Verification — 2026-11-13

- After the user changed direction back to self-written evaluation, reran the
  same targeted eval/training/LM command below: exit 0, `11 passed in 1.70s`.
  No new tests or production changes; no full-suite rerun for the scope change.
- After the user's `try/finally` change,
  `./test.sh tests/unit/core/test_trainer_evaluation.py` → exit 0,
  `4 passed in 1.22s`. The previously Red loss-error restoration case is Green.
- Added empty list/iterator cases, both requiring `ValueError` and restored
  training mode. Added a real training/evaluation scenario with two SGD momentum
  updates, snapshotting nonzero gradients, parameters, optimizer state, and
  progress. Evaluation preserves all snapshots exactly. No optimizer mocks or
  private state inspection are used.
- Updated the existing packed-token tiny LM integration to call public
  `Trainer.evaluate` before/after 30 training passes, replacing its batch-mean
  helper. It evaluates 64 targets each time, starts above loss 1, and finishes
  below 0.05, using real Adam updates.
- Targeted command:
  `./test.sh tests/unit/core/test_trainer_evaluation.py tests/integration/test_minimal_training_loop.py tests/integration/test_train_language_model_on_packed_tokens.py`
  → exit 0, `11 passed in 1.67s`.
- Full suite: `./test.sh` → exit 0, `99 passed in 1.38s`.
- The added preservation/empty-input cases and updated LM integration passed on
  their first run; no new Red is claimed for already supported behavior. No
  production/dependency/test-configuration changes were made by the agent.
- Next: prepare the fixed real dev subset and agree on its tokenization/packing
  settings, then run a manual evaluation with a real causal LM. Only after that
  agree on automatic scheduling/logging API. Flock integration, padding/masks,
  distributed reduction, mixed module modes, and GPU execution remain unverified.

## Earlier Mode-Restoration Red — 2026-11-13

- After the user's weighted aggregation/type/no-grad/mode changes, the original
  targeted command returned exit 0, `1 passed in 1.26s`.
- Added two normal-path cases (initial train/eval mode) and one loss-error case.
  A public forward hook observes eval mode, disabled gradient recording, and an
  output without a gradient path. The normal cases verify the caller's gradient
  context and prior mode are restored. No forward-call count is required.
- `./test.sh tests/unit/core/test_trainer_evaluation.py` → exit 1,
  `1 failed, 3 passed in 1.30s`.
- `./test.sh` → exit 1, `1 failed, 95 passed in 1.52s`.
- Sole failure:
  `test_evaluate_restores_training_mode_when_loss_computation_raises` at the final
  assertion: `model.training` is False, expected True. The original RuntimeError
  propagates unchanged, and the caller's gradient context is restored. Production
  restores the model mode only after the loop, so a loss error bypasses that line.
  This is the intended behavioral Red; the existing weighted-loss case stays Green.
- Next: user restores prior model mode in `finally`, then rerun the targeted cases
  and relevant regressions. Parameter/gradient/optimizer/progress preservation,
  empty input, data-iteration/forward failures, and LM integration remain untested.

## Earlier Verification — 2026-11-13

- Command: `./test.sh tests/unit/core/test_trainer_evaluation.py` → exit 1,
  `1 failed in 1.37s`. The full suite was not rerun for this check.
- The method now executes. Its first result assertion fails: `loss` is
  `tensor(1.6667, grad_fn=<DivBackward0>)`, not a Python float. Later numerical
  and target-count assertions have not executed.
- Source analysis: `total_loss += loss` sums batch means instead of summed
  target losses. The fixture therefore produces `(1 + 9) / 6` instead of
  `(4 * 1 + 2 * 9) / 6`. Use `loss.item() * targets.numel()` for each contribution.
  `total_num` starts at `0.0`, so the reported count is a float; initialize it as
  integer `0` for the agreed result type.
- Further source gaps: no no-grad context, and saved `prev_training` is never
  restored. These have not yet been covered by dedicated behavior tests. Restore
  mode in `finally` on both success and failure, assuming the agreed uniform mode.
- Next: user corrects aggregation and scalar/count types, then rerun the existing
  test. Add later lifecycle tests after this first case reaches Green.

## First Verified Red — 2026-11-12

Test: `tests/unit/core/test_trainer_evaluation.py::test_evaluate_weights_mean_loss_by_target_count_with_a_smaller_final_batch`.

- Real CPU float32 zero-weight linear model and mean MSE; no model/Trainer mocks.
  The first batch has two samples with two targets each (4 targets, loss 1).
  The final batch has one sample with two targets (2 targets, loss 9).
- Expected: Python float loss `(4 * 1 + 2 * 9) / 6 = 11/3` and Python integer
  `num_targets == 6`. Directly averaging batch means would incorrectly give 5;
  counting samples rather than targets would incorrectly report 3 targets.
- `./test.sh tests/unit/core/test_trainer_evaluation.py` → exit 1,
  `1 failed in 1.33s`.
- `./test.sh` → exit 1, `1 failed, 92 passed in 1.71s`.
- Observed: `AttributeError: 'Trainer' object has no attribute 'evaluate'` at the
  public method call. This is the intended missing-API Red, not an environment or
  reference failure. Result assertions have not executed yet. All existing tests
  remain Green.
- Next: user implements the agreed `evaluate` method, then rerun this test and
  relevant regressions. Mode/no-grad, state preservation, empty/error handling,
  and LM integration remain untested; they will follow in separate TDD steps.

After manual evaluation is Green, agree on optimizer-step-based scheduling,
logging at the training step, and timing treatment for training throughput.
Do not change `train()`'s current one-pass semantics merely to schedule evaluation.

## Final Flock Evaluation

The user's supplied requirements specify `fineweb_edu`, `base_easy`, and `paloma`,
with final commands using `--dtype float32`. Preserve Flock's metric names/units;
token perplexity and byte-normalized BPB are different quantities. Keep the
official evaluator outside the training core.

Before a long training run, perform a small export/load smoke check to establish
checkpoint, model configuration, tokenizer, and model-registration requirements.
Full final scoring can wait until the end. The documentation check below supersedes
the initial access failure. No installed Flock version or executable compatibility
has been verified.

## Official Documentation Verified — 2026-11-12

After web-tool access and sandbox DNS failed, public pages were fetched with curl
outside the sandbox and read locally. Their source links identify documentation
revision `710e75a857969be3fefef67c50b3df016f15ef15`. This is documentation evidence,
not a deployed version or an execution result. No package was installed, model
uploaded, evaluation run, or production/test code changed. The existing baseline
above was not rerun for this documentation-only check.

- [FineWeb-Edu guide](https://docs.cerulean.works/flock/main/guide/evals/fineweb-edu/):
  5,000 validation documents; tokenize text without special tokens; use BOS (EOS
  fallback) as an input-only prefix; no appended EOS target or cross-document
  packing. Score every text token once in disjoint rolling windows of 8192
  predicted tokens, with preceding context for the final partial window. Sum NLL
  across all documents and divide by total predicted tokens. Formal reporting
  uses FP32; BF16 is acceptable for training checks.
- **Context requirement:** `fineweb_edu` refuses a model with `max_length < 8192`.
  The lab's 4096 training sequence length is not itself a limit on model context,
  but the chosen architecture/configuration must actually support 8192 at eval.
  Do not substitute another task or merely change metadata. The documented
  `fineweb_en` uses the same documents with 2048 windows but is a different task.
- [Metrics](https://docs.cerulean.works/flock/main/reference/evals/metrics/):
  perplexity tasks report `loss_per_token`, `perplexity = exp(loss_per_token)`, and
  `bits_per_byte = total_nll / (ln(2) * total_utf8_bytes)`. Their aggregation is a
  ratio of sums. Multiple-choice `bpb` instead averages per-document scores.
- [Task catalog](https://docs.cerulean.works/flock/main/reference/evals/tasks/):
  `base_easy` is an unscored group containing QA accuracy, QA BPB, math BPB, and
  code BPB suites; the documented names are `base_easy:qa_rc`,
  `base_easy:qa_bpb`, `base_easy:math_bpb`, and `base_easy:code_bpb`. The catalog
  does not list the image's shorthand `base_easy:bpb`. `paloma` groups nine
  sources (309 domain tasks from the table), uses 2048-token windows, and has no
  overall aggregate; each domain's primary metric is `bits_per_byte`. The dataset
  is gated. Final required suite naming should follow the lab's selected version.
- [CLI](https://docs.cerulean.works/flock/main/reference/evals/commands/):
  `--task fineweb_edu`, `--suite base_easy`, `--suite paloma`, and
  `--dtype float32` are documented. `--limit N` evaluates the first N documents
  per task, not the full benchmark. Each output directory must be new/empty.
  HF model references require `repo@revision`; Flock registry models use
  `flock:org/name@version`. Default weight dtype is BF16, so final commands must
  set FP32 explicitly. Local scoring does not require tracker sign-in.
- [In-memory evaluation](https://docs.cerulean.works/flock/main/guide/evals/evaluate-checkpoint/)
  and [Python API](https://docs.cerulean.works/flock/main/reference/evals/python-api/):
  `ScoringModel(model, tokenizer, ...)`, `prepare_task(scorer, task, limit=N)`,
  and repeated `score_prepared(scorer, prepared)` permit limited evaluation
  without saving/reloading weights or repeating tokenization. Keep the same
  tokenizer object. Existing model dtype/device are used; gradient recording is
  disabled and prior per-module modes/SDPA settings are restored. Models must
  meet the scorer's causal-LM calling convention; Upwash's current tensor-return
  examples are not automatically compatible. The API returns metrics without
  automatically attaching them to a training run.
- [Model storage](https://docs.cerulean.works/flock/main/guide/hub/store-models/)
  and [evaluation loading](https://docs.cerulean.works/flock/main/guide/evals/run-evaluation/):
  HF pinned causal-LM checkpoints and Flock registry models are documented routes.
  Flock-native storage uses safetensors weights, frozen-dataclass `config.json`,
  and `flock.json` class/dtype/provenance metadata; registry evaluation needs
  `hf_repo` for its tokenizer. These requirements do not establish that an
  arbitrary Upwash `state_dict` is loadable or that a custom class is supported.

Planning implication: retain a small local evaluation as requested, but label its
4096 packed score as a development metric. Keep the option of limited official
in-memory evaluation for an early integration check and checkpoint comparisons.
Resolve actual model support for 8192 before the long training run. No Flock
dependency addition or automatic evaluation scheduling has been agreed.

Latest priority update — 2026-11-13: after exploring
[018 — Final Flock Evaluation](018-flock-final-evaluation.md), the user stopped
the Flock integration and requested a self-written evaluator. Reuse the existing
manual API. A scope question is pending: begin with FineWeb-Edu dev loss/PPL, or
reimplement all three benchmark families. Do not assume the user approved
reimplementing Base Easy prompts and Paloma aggregation. Real-data evaluation
and automatic scheduling remain pending. The lab image's stated requirement to
use Flock for official scoring is not automatically satisfied by a custom eval;
acceptance of an alternative has not been established.
