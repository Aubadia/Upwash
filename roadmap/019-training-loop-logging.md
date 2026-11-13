# 019 — Training Loop Metrics and Logging

## Status — 2026-11-13

Trainer now computes global L2 across detached parameter gradients after backward and before the optimizer
update, excluding parameters without gradients and preserving gradient values.
Both regular and final-partial events emit `train/grad_norm` for the last
completed update. Computation runs only when logging is enabled; conversion to
a Python float happens only on emission. The TODO comment was removed.

- `./test.sh tests/integration/test_training_logging.py` → exit 0,
  `6 passed in 1.35s`. The existing one-update norm case is Green.
- Full suite not rerun. No test, dependency, or configuration changes.

Next: add multi-step and final-partial norm assertions. Missing-gradient,
clipping/AMP, and multi-device behavior remain unverified. Earlier records follow.

Next metric Red: `train/grad_norm`, using the screenshot's existing metric key
and existing Builder/Trainer API. Baseline `./test.sh` → exit 0,
`104 passed in 1.80s`. Added
`test_built_trainer_logs_global_l2_gradient_norm_for_one_update` in
`tests/integration/test_training_logging.py`.

The real CPU float32 model has two weights and a bias, all initialized to zero.
Input `[3, 4]`, target 1, and mean MSE give weight gradients `[-6, -8]` and bias
gradient `-2`; global L2 is `sqrt(104)` (approximately 10.198039), not the sum
of the two parameter tensors' norms (12). The test checks actual SGD updates
and unchanged gradients, then requires a Python-float norm alongside loss/LR
in the single event at step 1. It fails specifically with
`KeyError: 'train/grad_norm'` after the update/gradient/event assertions pass.

- `./test.sh tests/integration/test_training_logging.py` → exit 1,
  `1 failed, 5 passed in 1.14s`.
- `./test.sh` → exit 1, `1 failed, 104 passed in 1.39s`.

Only tests/docs changed. Next: user adds global L2 norm logging without
mutating gradients, then rerun the targeted case. No clipping exists in this
scenario, so it does not establish before-clipping timing with actual clipping.
Using one update with interval 1 deliberately leaves multi-step sampling versus
aggregation undecided. Final-partial emission, missing gradients, clipping/AMP,
and multi-step norm semantics need subsequent cases/agreement.
Earlier status records are historical.

request. Trainer captures the first parameter group's LR as a Python float
before each optimizer update when logging is enabled, and includes the last
completed update's rate in both interval and final-partial loss events.
No configuration/API changes or other metrics were added.

- `./test.sh tests/integration/test_training_logging.py` → exit 0,
  `5 passed in 1.66s`.
- `./test.sh` → exit 0, `104 passed in 1.66s`.

The two previously Red single-group LR cases are Green. This increment changed
Trainer and documentation; existing tests were preserved. Scheduler changes,
multiple parameter-group reporting, and remote delivery remain unverified.
Next: agree on gradient norm sampling before adding its smallest test; proposed
global L2 across gradients on the last completed update, before clipping.
Token/throughput and automatic evaluation APIs remain pending as described below.
Earlier Red/status records are historical.

The user requested tests for all seven metrics in the supplied Logging and
Submission screenshot. Following the project's small-step TDD workflow, the
next Red covers the existing single-group `train/learning_rate` metric key;
no new constructor or `train()` API is introduced. The screenshot directly
specifies this key. Multi-group naming and the other pending policies below
are not assumed by this test.

Baseline `./test.sh` → exit 0, `102 passed in 1.88s`.
Added `test_built_trainer_logs_single_group_learning_rate_with_each_loss_event`
in `tests/integration/test_training_logging.py`, parameterized for LR 0.1/0.05.
It uses real CPU SGD and checks three updates (weights 0.976/0.542), then expects
the configured LR as a Python float alongside loss in both events at steps 2/3.
Both cases reach the payload assertion and fail with
`KeyError: 'train/learning_rate'`: the events currently contain only loss.

- `./test.sh tests/integration/test_training_logging.py` → exit 1,
  `2 failed, 3 passed in 1.46s`.
- `./test.sh` → exit 1, `2 failed, 102 passed in 1.68s`.

These are intended missing-metric Reds; existing tests remain Green. Only
tests/docs changed. Next: user includes the optimizer's single-group learning
rate in both regular and final-partial loss events, then rerun the test.
This constant-rate case does not verify scheduler changes or multiple groups.
After Green, agree on gradient norm sampling (proposed global L2 on the last
completed update, before clipping), group key names, token-count/clock policy,
and repeatable evaluation inputs/schedule before writing their tests.
Actual remote delivery is not tested here. Earlier status entries are historical.

Current verification: the user changed both Config and Trainer defaults to
`None`. Baseline `./test.sh` → exit 0, `100 passed in 1.91s`; the previous
default-related failures are resolved in the current files.

The existing logging integration is now parameterized for 2, 3, and 4 updates
at interval 2. Exact boundaries emit only at `[2]` or `[2, 4]`, with no duplicate
final event; the second full interval averages only its own losses (`1.343488`).
The original final partial interval case remains unchanged in meaning.
`./test.sh tests/integration/test_training_logging.py tests/integration/test_built_trainer.py tests/unit/test_training_config.py tests/unit/test_builder.py`
→ exit 0, `13 passed in 1.48s`. The new cases passed on first execution; no
artificial Red is claimed. The full suite was run before these test-only changes,
not after. Only tests/docs changed in this increment.

Next: verify that an explicit interval with no logger still permits real training
without logging. Then agree on interval validation before adding its tests.
No new metrics/API, production code, dependencies, or remote actions were added.
Earlier diagnostic results below are historical.

Latest follow-up: the user made the config interval optional with default `10`.
`./test.sh` → exit 1, `2 failed, 98 passed in 1.99s`. Construction now succeeds;
both built-Trainer update cases fail because they expect no automatic logging
when the setting is omitted, but default `10` enables logging and the final
partial-interval flush emits `train/loss=4.0` at step 1. Next: change the config
default to `None` (and align Trainer's default for direct callers), then rerun
the suite. No production code or tests changed in this diagnostic follow-up.

Follow-up full-suite verification: `./test.sh` → exit 1,
`5 failed, 95 passed in 1.86s`. All five failures occur at construction because
`TrainingConfig` now requires `logging_steps`, while existing callers omit it.
Affected cases: both built-Trainer update variants, both Builder unit cases, and
the optimizer-config composition case. The interval logging integration passes.
Expected compatibility: omission constructs successfully and disables automatic
logging. Next: user gives `TrainingConfig.logging_steps` a default of `None`;
align Trainer's default with `None` for direct callers, then rerun `./test.sh`.
No production code or tests were changed in this diagnostic follow-up.

At the user's explicit request to implement Trainer's logging TODO, the agent
added interval mean `train/loss` events after completed updates and a final
partial-interval flush. Detached losses accumulate without per-step `.item()`
calls; conversion to a Python float happens only when emitting an event.
Logging is inactive when the logger or interval is `None`.

Before this change, `./test.sh tests/integration/test_training_logging.py`
failed because no events were emitted (`[]` instead of steps `[2, 3]`); config
and Builder wiring now reach the loop successfully. After implementation,
`./test.sh tests/integration/test_training_logging.py tests/unit/core/test_trainer_loss.py tests/unit/core/test_trainer_empty_batches.py tests/unit/core/test_trainer_evaluation.py`
passed: `12 passed in 1.54s`. The full suite was not run. Existing constructor
defaults were preserved: Trainer defaults to 10 and TrainingConfig requires an
interval, so the agreed omitted-setting compatibility remains unfinished.
Next: restore omitted-setting compatibility, then test exact-interval completion
and disabled logging before extending metrics. The older Red below is historical.

The user prioritizes training-loop logging, supplying the lab's Logging and
Submission requirements. They selected `logging_steps` on `TrainingConfig`,
wired through Builder, rather than a new argument on `train()`.

Branch `feat/Trainer-Log`, HEAD `6476e04` (`feat: minimum evaluation`). The worktree
was clean before this task. Baseline `./test.sh` → exit 0,
`99 passed in 2.07s`. The first new integration case is now Red for the missing
`TrainingConfig(logging_steps=...)` argument. The agent changed only tests/docs.

## Agreed First Increment

```python
config = TrainingConfig(
    optimizer=OptimizerConfig(lr=0.1),
    logging_steps=2,
)
trainer = build_trainer(
    config=config,
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=torch.optim.SGD,
    logger=logger,
)
state = trainer.train(batches)
```

- Store the logging interval on configuration; Builder wires it into Trainer.
  The precise internal wiring does not need a new shared abstraction.
- Count completed optimizer updates. Emit an event with `train/loss` after every
  `logging_steps` updates; flush an incomplete final interval on normal completion.
  Do not duplicate an event when training ends on an exact interval boundary.
- The interval metric is the arithmetic mean of the supplied scalar step losses,
  each computed before that step's update. This is distinct from `evaluate`'s
  target-weighted aggregation. Gradient accumulation is not part of this increment.
- Log detached Python numbers through the supplied logger; Trainer stays independent
  of MemoryLogger/FlockLogger implementations.
- Omitting `logging_steps` keeps automatic logging disabled (proposed default
  `None`), preserving the existing built-Trainer test and legacy callers.
- Preserve `state.global_step`'s agreed per-call update count. A proposed
  `step_offset` was not adopted as a `train` argument; cumulative remote steps
  across repeated calls must be agreed separately before remote delivery testing.
- Validation of interval values and behavior with enabled logging but no logger
  remain to be specified/tested in subsequent small steps.

## First Verified Red — 2026-11-13

Test:
`tests/integration/test_training_logging.py::test_built_trainer_logs_interval_mean_loss_and_the_final_partial_interval`.

The real CPU float32 model is `prediction = w*x`, initialized at `w=0`, with
`x=1`, `target=2`, mean MSE, and real SGD at learning rate 0.1. Three updates
produce pre-update losses 4, 2.56, and 1.6384. Required records at interval 2:

| Completed step | `train/loss` |
| --- | --- |
| 2 | `(4 + 2.56) / 2 = 3.28` |
| 3 | `1.6384` (final partial interval) |

The test uses the real Builder, Trainer, optimizer, and MemoryLogger. It permits
additional metric keys so later increments can extend events without weakening
the loss/count/timing expectations.

- `./test.sh tests/integration/test_training_logging.py` → exit 1,
  `1 failed in 0.92s`.
- `./test.sh` → exit 1, `1 failed, 99 passed in 1.83s`.
- Observed: `TypeError: TrainingConfig.__init__() got an unexpected keyword
  argument 'logging_steps'`. This is the intended missing-API Red, not an
  environment failure. Construction stops before training/logging assertions.
  All existing tests remain Green.
- Next: user accepts/stores the config setting, connects it through Builder to
  Trainer, and implements interval loss emission plus final partial-interval
  flushing. Then rerun the targeted test before adding another metric.

## Remaining Lab Metrics — Planned, Not Implemented/Tested

The screenshot specifies project `lab6-predictable-scaling`. Keep project/run
initialization with the caller and existing logger. No new remote logger is
required, and no upload/run creation is part of this local TDD step.

| Metric | Required meaning and next design point |
| --- | --- |
| `train/loss` | Mean step loss in the logging interval; first increment above. |
| `train/grad_norm` | Global gradient norm before clipping. Proposed L2 norm across parameters with gradients, measured on the logged step after backward; clipping/AMP semantics must follow their actual implementations. |
| `train/learning_rate` | Rate used for the completed update. Log separate parameter-group keys when rates differ; group naming still needs agreement. |
| `train/tokens_per_second` | Proposed interval non-padding training tokens divided by interval training time. Define timing boundaries and exclude evaluation; GPU synchronization/measurement needs its own verification. |
| `train/total_tokens_seen` | Cumulative non-padding input tokens processed, excluding eval. In unpadded `[B, T]` LM inputs this is `inputs.numel()`. Generic MSE/regression inputs are not automatically tokens; counting/masking needs an explicit LM policy. |
| `eval/loss` | Token-weighted natural-log cross-entropy on the designated dev set; existing manual evaluator supports dense unpadded mean-loss batches. Real dev data and automatic scheduling remain pending. |
| `eval/perplexity` | `exp(eval/loss)` for next-token cross-entropy, not arbitrary supplied losses. |

Suggested sequence after interval loss: learning rate and gradient norm; token
count/throughput; then scheduled dev evaluation and its two metrics. Agree on
each new interface before testing it. Use one metric event at a shared step when
train/eval schedules coincide, so an SDK commit does not split same-step values.

Evaluation must occur at least once per 10% of planned optimizer updates. A
proposed integer interval is at most `max(1, planned_steps // 10)`; record a final
evaluation as well. This is independent of training metric `logging_steps`.
The data must be repeatable for successive evaluation calls; a consumed one-shot
iterator is unsuitable. No eval schedule API is agreed yet.

## Experiment Configuration and Submission Context

The supplied requirements also call for model configuration, initialization,
optimizer settings, batch size, schedule, and training duration in the run config.
The caller owns these experiment settings; existing logger config forwarding
does not establish that a real run contains them. Planned duration/token budget
and actual elapsed time need honest separate recording.

The final recipe/configuration and forecast are submitted before the final run;
the designated endpoint is 200B training tokens, and final weights/tokenizer are
uploaded to Hugging Face with `eval_model_id` set in the OJ script. These are
recorded requirements, not actions authorized or executed by this task. No
credentials, services, dependencies, or remote state were changed.
