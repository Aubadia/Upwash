# Implementation Progress

Last updated: 2026-11-14 (Asia/Taipei).

Merge verification: merged `origin/main` at `5d989c7` into `feat/Adaptor`.
Resolved the progress-log conflict by retaining both branches' records. Adapted
the incoming Builder precision test to the existing model/config SGD test
factory. Initial `./test.sh --tb=short`: exit 1, `1 failed, 317 passed`
because direct SGD rejected the factory's `config` keyword. After adapting the
test: exit 0, `318 passed in 3.66s`; `git diff --check` passes. No current
merge blocker. Next for task 020's HF adapter: tiny real HF Builder/Trainer
integration. Both branches use task number 020; links identify their distinct
documents, and renumbering remains deferred.

Lab requirements are consolidated in [docs/Spec.md](../docs/Spec.md), including
compute grants, exact token accounting, architecture, evaluation, forecasts,
submission, and unresolved source inconsistencies. This documentation update
adds no API or implementation. Session baseline: `./test.sh` → exit 0,
`276 passed in 4.67s`. Current priority remains task 019: agree on explicit
LM perplexity opt-in before writing its first test. External scoring units,
grant counts, and the official final configuration remain clarification items
before launching the affected runs.

## Resume Here

Current task (2026-11-13): [021 — Eval perplexity](021-eval-perplexity.md), on branch `feat/eval-perplexity` (PR #11).

- Agreed interface: `Trainer(..., eval_perplexity=False)` and `build_trainer(..., eval_perplexity=False)`.
  - When enabled, every event carrying `eval/loss` also carries `eval/perplexity = exp(eval/loss)`.
  - Overflow gives `inf`, and training continues.
- Red verified: `5 failed, 4 passed` on the targeted files, all `TypeError` for `eval_perplexity`. The rest of the suite gives `296 passed`.
- Sanity check: a scratch reference implementation passed the full suite (`302 passed`).
- Green (2026-11-14): the user implemented it, and the full suite returned `302 passed`. The internal dev data was prepared with `scripts/prepare_dev.py`.
- 2026-11-14: merged `origin/main` at `9940107` (PR #10) into `feat/eval-perplexity`.
  - The textual conflict in this file was resolved by keeping both branches' records.
  - The eval-perplexity builder test was adapted to PR #10's `factory(model, config=...)` contract with the `sgd_optimizer_factory` fixture.
- Next for evaluation:
  - Internal dev data: a script that tokenizes `cerulean-works/fineweb-edu-dev` as `[BOS, text, EOS]`, and an eval `DataLoader` in the entry script.
  - Flock `fineweb_edu` scoring after training. This needs the TA's answer on 8192, a Flock environment, an α-folding HF conversion, a scoring script, and a tiny end-to-end smoke run.
- Evaluation scope agreed with the teammate:
  - Internal fineweb-edu dev PPL during training. The dev data is `cerulean-works/fineweb-edu-dev`, prepared as `[BOS, text, EOS]` with the existing `prepare_token_shards`.
  - Flock `fineweb_edu` scoring after training to compare hyperparameters.
  - The teammate reports that the training data excludes the 5,000 dev documents.
- Open: the Flock `fineweb_edu` task refuses a model with `max_length < 8192`, re-read in the Flock docs on 2026-11-13, while training uses context 4096. Ask the TA how these fit.

Teammate PR #10 (`feat/Adaptor`: HF Llama adaptor, Muon + AdamW, WSD scheduler), reviewed 2026-11-14:

- Tests (Docker, `upwash-venv-pr5`, after `uv sync` installed `transformers==5.22.2` with the user's approval):
  - PR #10 alone: `318 passed`.
  - Merged with PR #11: `1 failed, 323 passed`. `test_build_trainer_passes_eval_perplexity_to_the_trainer` passes `torch.optim.SGD`, but PR #10 changes the factory contract to `factory(model, config=...)`, which gives `TypeError: SGD.__init__() got an unexpected keyword argument 'config'`. PR #10 merged first, and PR #11 now uses the `sgd_optimizer_factory` fixture.
  - Only `roadmap/README.md` conflicts textually.
- Its roadmap files 020–023 collide with merged 020 (device and precision) and PR #11's 021.
- Fixes deferred by the user, who wants things running first:
  - Option-B initialization and α are not applied, although `ModelConfig` records them. This must be fixed before any recorded run.
  - `max_position_embeddings` is 4096, which is pending the host's answer on the Flock 8192 requirement.
  - Muon/AdamW grouping is by parameter name.
  - The WSD `"linear"` branch is unreachable.
  - `train/learning_rate` logs only the Muon group.
  - The model config doesn't set `use_cache=False`.

Teammate's records from PR #10 (`feat/Adaptor`):

Latest user-directed work: [020 — HF Llama Loss Adapter](020-hf-llama-adaptor.md).
Latest verification is Green: user's causal_lm_loss implementation passes
the hand-calculated no-shift example and real HF reference loss/gradient
comparison. Targeted adaptor/core-loss tests: `6 passed in 2.82s`; full
`./test.sh --tb=short`: `298 passed in 3.39s` (exit 0). No current blocker.
Next: real tiny HF Llama Builder/Trainer training/evaluation with Muon/AdamW
and WSD; custom initialization/output scaling remain separate. Agent changed
documentation only in this check. Earlier Red below is historical.

Approved `causal_lm_loss(outputs, targets)` in adaptor.py: mean token CE over
outputs.logits [B,T,V] against already-shifted targets [B,T], with no second
shift and preserved gradients. Added a hand-calculated example and a real HF
Llama loss/gradient comparison; reference torch 2.15.0+cu132/transformers 5.22.2
forward/backward succeeds first. Targeted adaptor/core-loss tests:
`2 failed, 4 passed in 5.18s`; full `./test.sh --tb=short`:
`2 failed, 296 passed in 3.78s` (exit 1). Both intended Reds are the missing
causal_lm_loss import; Upwash numerical assertions have not executed. Next:
user implements the adapter, then verify Green before full HF training
integration. Tests/docs only; no environment blocker or production changes.

Latest user-directed work: [023 — LR Scheduler Wiring](023-lr-scheduler.md).
Latest increment is Green: WSD unit coverage now has 13 explicit examples for
linear/cosine/nonlinear-power schedules and zero-length/fractional phase
boundaries, using real Muon/AdamW groups and independently calculated expected
values. New cases passed immediately. Targeted scheduler/config/training tests:
`47 passed in 2.41s`; full `./test.sh --tb=short`: `296 passed in 3.38s`
(exit 0). Tests/docs only; no current blocker. Next: agree on the next training
increment, such as HF output adaptation; scheduler persistence, broader runtime
edge guarantees, per-group LR logging and GPU/bf16 remain separate work.
Earlier results below are historical.

Latest increment is Green: added real WSD + Muon/AdamW + Builder/Trainer
10-update integration coverage. Both LR groups follow the agreed factors and
retain their ratio; final parameters match independent real torch references
using manually assigned LRs. The zero-LR final update leaves all parameters
unchanged; logged first-group LRs describe the updates actually used.
Targeted tests: `4 passed in 2.49s`; full `./test.sh --tb=short`:
`284 passed in 3.64s` (exit 0). The test passed immediately, with no new Red
or production changes in this increment. No current blocker. Next: numerical
cosine/nonlinear-power and phase-boundary coverage, then HF output adaptation
and device/bf16 milestones. Earlier Red records below are historical.

Current increment: user approved WSD rounding/endpoints. Added the agreed
10-update linear schedule test for real Muon/AdamW groups with base LRs
0.008/0.0128, checking common factors and mutation of original groups.
Targeted scheduler/config/timing tests: `1 failed, 32 passed in 2.52s`;
full `./test.sh --tb=short`: `1 failed, 282 passed in 3.43s` (exit 1).
Sole intended Red: missing `upwash.components.schedulers` at the
`build_wsd_scheduler` import; both child optimizers construct successfully.
Numerical schedule assertions have not executed. Next: user implements
`components/schedulers/wsd.py`, then verify Green. No production changes or
environment blockers. Boundary/nonlinear schedule cases remain later tests.
Earlier results below are historical.

Latest verification is Green: Trainer advances its optional lr_scheduler
after optimizer.step(). Timing/used-LR test now passes all assertions.
Targeted tests: `7 passed in 2.24s`; full `./test.sh --tb=short`:
`282 passed in 3.40s` (exit 0). Next: confirm WSD rounding and first/final
update conventions before its first numerical test. No concrete scheduler
exists yet; proposed multipliers are recorded in task 023. No production/test
changes in this verification step. Earlier results below are historical.

Current increment: Builder wiring is Green (`4 passed in 2.33s`). Added a
real-SGD two-update integration test with a halving scheduler test double.
Expected used LRs 0.1/0.05 yield final weight 0.56; actual is approximately
0.72 because Trainer never advances its injected scheduler. Targeted tests:
`1 failed, 6 passed in 2.50s`; full `./test.sh --tb=short`:
`1 failed, 281 passed in 3.32s` (exit 1). Next: user calls optional
`lr_scheduler.step()` after optimizer.step(), retaining logs of used LR, then
verify Green. Scheduler count/final LR/log assertions remain behind the weight
failure; WSD math is not tested yet. Agent changed tests/docs only.
Earlier results below are historical.

Newest verification: optional defaults are fixed; all 48 old regressions are
resolved. Targeted tests: `1 failed, 35 passed in 2.33s`; full
`./test.sh --tb=short`: `1 failed, 280 passed in 3.43s` (exit 1).
Sole Red is Builder rejecting `schedule_config`; it still needs that input,
`total_steps`, and the factory call with optimizer/config/total_steps.
No production/test changes during verification. Earlier results below are
historical.

Full-suite recheck after the user's report: `./test.sh --tb=short` → exit 1,
`49 failed, 232 passed in 3.67s`. The 48 old failures come from required
`lr_scheduler`/`lr_scheduler_factory` inputs; the new test fails on missing
`schedule_config`. No numerical training/scheduler assertions execute in these
cases. Next: restore optional `None` defaults, add Builder schedule inputs and
forward optimizer/config/total_steps. Documentation-only verification step.

User corrected the Trainer argument/attribute to `lr_scheduler`; the test now
expects `trainer.lr_scheduler`. Builder factory remains `lr_scheduler_factory`.
Trainer now saves the value. Earlier naming discussion below is historical;
remaining work is optional defaults and passing optimizer/config/total_steps
to the factory.

Latest targeted recheck: `5 failed, 31 passed in 2.29s` (exit 1).
Builder rejects `schedule_config` and requires `lr_scheduler_factory`, breaking
four existing callers. Source also calls the scheduler factory without its
required inputs; Trainer requires `lr_scheduler` but does not store it. Tests
still expect the earlier agreed `trainer.scheduler` name. Next: complete
optional defaults, factory arguments and Trainer storage; reconcile naming
before changing expectations. No full-suite run or production/test changes.
The first Red result below is historical.

User chose Builder inputs `lr_scheduler_factory`, `schedule_config`, and
`total_steps`, with no `partial` required. First test expects the factory to
receive the constructed optimizer plus settings, and its result to be stored
as `trainer.lr_scheduler`, without advancing training/scheduling during construction.
`./test.sh tests/unit/test_builder.py tests/unit/test_schedule_config.py tests/integration/test_muon_adamw_trainer.py --tb=short`
→ exit 1, `1 failed, 35 passed in 4.24s`. Sole intended Red: Builder rejects
the new `lr_scheduler_factory` keyword. No environment blocker or production
edits. Next: user implements optional Builder inputs and Trainer scheduler
storage, then verify Green. Update timing and WSD mathematics remain later
tests; full suite not rerun after this new test.

Latest user-directed work: [022 — Muon and AdamW Factory](022-muon-adamw-factory.md).
Builder integration is now Green. The factory/CombinedOptimizer pass the
original group/config test. Added a real CPU tiny-LM two-batch training test
using `partial(build_muon_adamw, model_config=model_config)` with the existing
Builder; parameters and last-batch gradients match independent torch Muon and
AdamW references. Bias and tied-embedding updates are covered. Targeted tests:
`5 passed in 3.08s`; full `./test.sh --tb=short`: `280 passed in 3.42s`
(exit 0). No production changes or new Red in this step. No current blocker.
Next: agree on further behavior; direct HF output adaptation, optimizer edge
cases, schedules/checkpoints and per-group LR logging remain pending.
Earlier implementation and Red records below are historical.

Approved `build_muon_adamw(model, *, config, model_config)` under
`components/optimizers/muon_adamw.py`, using `ModelConfig.d_model` for decay
scaling. Attention/MLP matrices go to Muon; embedding/head/norm/bias to AdamW;
tied weights are assigned once. First test uses a real tiny tied HF Llama
and real torch 2.15.0+cu132 reference optimizers, which construct successfully.
Targeted factory/Builder/adaptor tests: `1 failed, 4 passed in 3.84s`;
full `./test.sh --tb=short`: `1 failed, 278 passed in 3.55s` (both exit 1).
Sole intended Red is the missing `upwash.components` module at the agreed
factory import; Upwash group assertions have not executed. No environment
blocker or production edits. Next: user implements factory/combined optimizer,
then verify Green before update and gradient-clearing increments.

Latest user-directed work: [021 — Model-Aware Optimizer Factory](021-optimizer-factory.md).
Latest verification is Green: user implemented
`optimizer_factory(model, config=config.optimizer)`. Real Muon factory test
passes. Legacy SGD test callers now use a shared test-only adapter, retaining
all numerical expectations. Targeted Builder/config/logging/evaluation tests:
`72 passed in 2.37s`; full `./test.sh`: `278 passed in 3.30s` (both exit 0).
No current blocker. This completes factory injection; next agree on the
production Muon/AdamW factory and parameter grouping before writing its tests.
The current Muon factory is test-local. Agent changed tests/docs only.
Earlier failure records below are historical.

Newest check: the single Muon factory test still fails because Builder passes
`model.parameters()` rather than `model`. The `config=` keyword is fixed.
Command: `./test.sh tests/unit/test_builder.py::test_build_trainer_passes_model_and_complete_config_to_muon_factory --tb=short`.
Next: user passes the complete model; then migrate legacy SGD test callers and
verify Green. No broader tests or production edits in this check.

Latest implementation check: the targeted Builder/config command below now
returns `5 failed, 43 passed in 1.55s` (exit 1). Builder passes
`model.parameters(), lr=config.optimizer`: Muon factory rejects `lr`, while
old SGD callers receive a config object as their numerical learning rate.
Next: user changes the call to `optimizer_factory(model, config=config.optimizer)`;
then adapt legacy SGD test callers. Full suite not rerun. The original Red
result below is historical.

Agreed contract: `optimizer_factory(model, *, config: OptimizerConfig)`;
the user requested real Muon in the test. A standalone CPU Muon gradient
update succeeds with torch 2.15.0+cu132. New Builder test is intended Red:
`./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py tests/unit/test_training_config.py`
→ exit 1, `1 failed, 47 passed in 2.16s`. Builder still passes `lr=` instead
of the complete config/model; factory assertions have not executed. No
environment blocker or production changes. Next: user changes the Builder
call; adapt old SGD test callers to the approved contract and verify Green.
Combined Muon/AdamW parameter grouping and width-scaled decay remain later
increments. Full suite not rerun.

Latest user-directed work: [020 — HF Llama Adaptor](020-hf-llama-adaptor.md).
Implemented the config adaptor and corrected
the factory name to `build_llama_model`. Latest verification:
`./test.sh tests/unit/test_adaptor.py tests/unit/test_model_config.py` → exit 0,
`58 passed in 2.76s`. CPU construction and the 4,432-parameter count are Green;
the full suite was not rerun. No current blocker. Next: agree on initialization,
output scaling and detailed forward tests; custom initialization/output scaling
remain unimplemented. The previous Red record below is historical.

The user approved `build_llama_model(ModelConfig)` returning a CPU
`LlamaForCausalLM`; `adaptor.py` is empty. The first test checks type, parameter
devices and a hand-calculated 4,432-parameter tiny model. The user installed
transformers, locked at 5.22.2 (torch 2.15.0). Real HF reference construction
succeeded on CPU with exactly 4,432 parameters. Targeted adaptor/config command:
`1 failed, 57 passed in 2.56s`; full `./test.sh`: `1 failed, 276 passed in 3.50s`.
Sole failure is the missing `build_llama_model` import, the intended API Red;
Upwash model assertions have not executed. The earlier dependency blocker is resolved.
Baseline recheck: model config `57 passed`; full `./test.sh` → exit 0,
`276 passed in 4.69s`. Next: user implements the configured CPU model factory,
then verify Green. Initialization/output scaling need later tests. Logging work below
remains pending separately.

Device/precision work merged from `origin/main` (the following records describe that branch):

Task (2026-11-13): [020 — Device placement and bf16 mixed precision](020-device-precision.md), on branch `feat/gpu-bf16` from `main` at `c37845f`. Merged through PR #9.

- Agreed interface:
  - Trainer moves each batch to the model's parameter device, in both `train` and `evaluate`. There is no `device` argument.
  - `precision` is `"fp32"` (default, no autocast) or `"bf16"` (autocast around `compute_loss`, float32 parameters). Trainer raises `ValueError` for any other value.
  - `TrainingConfig(..., precision="fp32")` passes through `build_trainer`.
- Red verified: `17 failed, 46 passed` on the targeted files, and the rest of the suite gives `276 passed`.
  - Device cases fail because batches reach a meta model unmoved.
  - Precision cases fail with `TypeError` for `precision`.
- Sanity check: a scratch reference implementation passed the full suite (`294 passed`).
- Review found a dtype-less `torch.autocast`, which defaults to float16 on CUDA. Added a regression test that sets the CPU autocast default to float16, and confirmed Red.
- Green: the user added validation and `dtype=torch.bfloat16`, and the full suite returned `296 passed`.
- Real CUDA execution is not yet verified. Check it in the first nano4 smoke run.
- Next: commit. GPU timing via `synchronize` and a CUDA-only test marker are deferred.

Numbering (decided 2026-11-13): tasks continue after 019, so device and precision is 020. No number is reserved for the Muon + AdamW builder; it takes the next free number when it starts. Earlier records that call it "task 020" predate this decision. Eval perplexity took 021. PR #10 then added 020–023 (HF adaptor, optimizer factory, Muon/AdamW, LR scheduler). Its 020 collides with device and precision, and its 021 with eval perplexity. Together with the earlier duplicated 017–019, renumbering is still pending agreement with the teammate.

Logging work, previous priority: [019 — Training Loop Logging](019-training-loop-logging.md).
Merge verification: integrated `origin/main` at `fccd286` into
`feat/Trainer-Log`. Resolved config/Builder conflicts by retaining the required
Muon/AdamW fields, data/model/schedule configs and dataset builder, plus Trainer
logging/token/evaluation wiring. The generic optimizer factory receives only
`lr`; logging/evaluation tests use the full optimizer-config fixture.
`./test.sh` → exit 0, `276 passed in 2.21s`. Trainer's refactor is unchanged.
Next: explicit LM perplexity opt-in as described in the logging task.

Automatic `eval/loss` is implemented: config/Builder/Trainer wire `eval_steps`,
and `train(..., eval_batches=...)` runs interval/final evaluations with no
duplicate final checkpoint. Coincident and final train/eval metrics share an
event; eval-only and no-logger runs are supported. Differing schedules exclude
evaluation/logging time from throughput. Existing defaults remain optional.
Targeted scheduled-eval/logging/manual-eval command → exit 0,
`30 passed in 1.60s`; full `./test.sh` → exit 0, `123 passed in 1.43s`.
Next: agree on explicit LM perplexity opt-in and add its first numerical test.
Interval validation, reusable-dev enforcement, real dev data, GPU/distributed
behavior, and remote delivery remain unverified. See 019 for detailed evidence.
Earlier records below are historical.

Automatic `eval/loss` API is agreed: config `eval_steps=None`
passes through Builder to Trainer; `train(..., eval_batches=dev_loader)` uses
a reusable dev iterable every N updates and at completion, with no duplicate
final evaluation. Coincident train/eval metrics share an event, and evaluation
is excluded from throughput. Perplexity follows with explicit LM opt-in.
Added two real-SGD integration cases for exact/final-partial evaluation at
interval 2. Expected post-update weighted dev MSE: 1.7850666667/1.3659093333;
train/eval metrics share events at steps [2] or [2, 3], without duplicate eval.
Targeted scheduled-eval/logging command → `2 failed, 18 passed in 1.73s`;
full `./test.sh` → `2 failed, 117 passed in 1.51s` (both exit 1).
Intended Red: Config rejects eval_steps, before runtime assertions execute.
Only tests/docs changed. Next: wire config/Builder/Trainer and add scheduled
and final evaluation, then verify Green. Details and remaining cases are in 019.

Throughput is implemented in both event paths. Interval tokens/time reset
after logging and at the start of each training call; final-partial timing
uses the last update's timestamp. Deterministic CPU cases verify padding,
all-padding rates, loading/forward/backward/SGD inclusion, logger/exhaustion
delay exclusion, repeated calls, evaluation gaps, and zero elapsed time.
Default/explicit disabled counting omits throughput.
`./test.sh tests/integration/test_training_logging.py` → exit 0,
`18 passed in 1.61s`; full `./test.sh` → exit 0, `117 passed in 1.47s`.
The original throughput case is Green. Next: agree on automatic evaluation
inputs and scheduling before adding eval/loss and eval/perplexity emission.
GPU/distributed timing and remote delivery remain unverified.
Earlier records below are historical.

Throughput timing is agreed: use existing `count_tokens=True`
to emit interval non-padding tokens per elapsed second, including data loading
and training computation, excluding logging, evaluation, and gaps between
training calls. Complete/final-partial intervals use `time.perf_counter`.
Added a deterministic CPU test: batches cost 5 seconds each (1 loading,
2 forward, 1 backward, 1 SGD), logger calls cost 100 seconds. For 8/8/4 tokens,
expected interval rates are 1.6/0.8 at steps 2/3. Training/update/count checks
pass, then lookup raises `KeyError: 'train/tokens_per_second'`.
Targeted logging tests → `1 failed, 14 passed in 1.40s`;
full `./test.sh` → `1 failed, 113 passed in 1.41s` (both exit 1).
Only tests/docs changed. Next: implement interval timing and emit throughput
in both logging paths, then verify Green. Remaining timing cases are in 019.

Token counting is implemented in Builder and Trainer with defaults
`count_tokens=False`, `pad_token_id=None`. Enabled counting sums input tokens
across training calls, excludes configured padding ids and evaluation, and
continues while logging is disabled. Both event paths emit Python integers.
Added coverage for padding ids 0/1, all-padding inputs, explicit/default
disabled counting, repeated calls, evaluation exclusion, and logger omission.
Targeted logging tests → exit 0, `14 passed in 1.40s`;
full `./test.sh` → exit 0, `113 passed in 1.50s`.
Next: agree on throughput timing boundaries and add a deterministic-clock test.
Throughput, scheduled evaluation, GPU behavior, and remote delivery remain
unverified. Earlier records below are historical.

Previous Red: `count_tokens=True` with optional
`pad_token_id`, replacing the callback proposal. Builder/Trainer default to
disabled counting and no padding id. Added the first unpadded LM integration
test: 8/8/4 input tokens must emit integer totals 16/20 at steps 2/3, with
real cross-entropy/SGD updates. Baseline: `107 passed in 2.12s`.
Targeted logging tests: `1 failed, 8 passed in 0.99s`; full `./test.sh`:
`1 failed, 107 passed in 1.32s`. Intended Red: Builder rejects `count_tokens`;
new training/payload assertions have not executed. Only tests/docs changed.
Historical next action: add/wire the optional settings and log token totals in both
event paths, then verify Green. Padding, lifetime cumulative counting,
evaluation exclusion, disabled-count compatibility, and throughput follow.

Latest verification: parameterized the one-update global L2 case to include an
unused trainable parameter and a frozen parameter. Each stays at `100`, retains
`grad is None`, and does not change the expected norm `sqrt(104)` or real SGD
updates. `./test.sh tests/integration/test_training_logging.py` → exit 0,
`8 passed in 1.46s`; full `./test.sh` → exit 0, `107 passed in 1.39s`.
Both new cases passed on first execution; no Red is claimed. Only tests/docs
changed. Next: agree on explicit LM token counting and interval timing before
testing `train/total_tokens_seen` and `train/tokens_per_second`. No new API is
agreed; generic regression inputs must not automatically count as tokens.

Latest verification: extended the existing 2/3/4-update integration cases to
assert last-update gradient norms at interval boundaries and the final partial
interval, plus unchanged final gradients and correct SGD weights. Expected
norms at steps 2/3/4 are `3.2`, `2.56`, and `2.048`.
`./test.sh tests/integration/test_training_logging.py` → exit 0,
`6 passed in 1.55s`; full `./test.sh` → exit 0, `105 passed in 1.37s`.
All added assertions passed on first execution; no Red is claimed. Only
tests/docs changed. Next: cover parameters without gradients, then agree on
token-count/throughput policy before adding new metric tests.

Latest implementation: Trainer computes global
gradient L2 after backward and before the optimizer update, leaving gradients
unchanged. Both interval and final-partial events include `train/grad_norm`
from the last update. Scalar conversion occurs only when emitting an event.
`./test.sh tests/integration/test_training_logging.py` → exit 0,
`6 passed in 1.35s`. Full suite not rerun. Next: add multi-step/final-partial
norm assertions; clipping/AMP and remote delivery remain unverified.
Earlier records below are historical.

Latest increment: added the one-update global L2 `train/grad_norm` test using
real SGD, two weights, and a bias. Expected `sqrt(104)` from gradients
`[-6, -8, -2]`; update/gradient/loss/LR assertions pass, then the metric lookup
raises `KeyError: 'train/grad_norm'`. Baseline: `104 passed in 1.80s`.
Targeted logging tests: `1 failed, 5 passed in 1.14s`; full `./test.sh`:
`1 failed, 104 passed in 1.39s`. Only tests/docs changed.
Next: user logs the global L2 norm without mutating gradients, then verify Green.
Multi-step norm sampling, actual clipping timing, and final-partial norm logging
are not established by this one-update case. Earlier records are historical.

Latest: at the user's explicit implementation request, Trainer now captures
single-group LR before optimizer updates and logs `train/learning_rate` beside
loss in both interval and final-partial events. Targeted logging test:
`5 passed in 1.66s`; full `./test.sh`: `104 passed in 1.66s`.
Loss and single-group constant-LR logging are Green. Next: agree on gradient
norm sampling (proposed global L2 before clipping on the last completed update),
then add its smallest test. Multi-group LR naming, token/throughput policies,
automatic evaluation inputs/schedule, and remote delivery remain pending.
This implementation increment changed Trainer/docs; earlier results follow.

The user requested coverage of the screenshot's seven metrics. The next small
increment is single-group `train/learning_rate` in the existing loss events,
including the final partial interval. Baseline: `102 passed in 1.88s`.
New real-SGD cases (LR 0.1/0.05) confirm updates and steps 2/3, then fail with
`KeyError: 'train/learning_rate'`. Targeted logging test: `2 failed, 3 passed
in 1.46s`; full `./test.sh`: `2 failed, 102 passed in 1.68s`.
Only tests/docs changed. Next: user adds LR to both event paths, then verify
Green. Gradient norm, multi-group keys, token/throughput policy, and automatic
eval inputs/schedule still require agreement before tests. Loss logging is
Green; this does not establish logging of the other metrics or remote delivery.
Earlier status/next-action entries below are historical.

Current files default both Config and Trainer intervals to `None`; baseline
`./test.sh` is Green: `100 passed in 1.91s`. Added exact-interval cases (2 and
4 updates at interval 2) alongside the existing partial-interval case. All pass
on first execution: logging/Builder/config regression command →
`13 passed in 1.48s`. No artificial Red; the full suite was not rerun after
these test-only changes. Only tests/docs changed in this increment.
Next: verify training with an explicit interval and no logger, then agree on
interval validation. Earlier failures and next-action entries below are historical.

Latest full suite after the user set the config default to `10`:
`./test.sh` → `2 failed, 98 passed in 1.99s`. Both built-Trainer cases expect
disabled logging when omitted, but the default enables the final partial flush.
Next: default the interval to `None` in config and Trainer, then rerun the suite.
This diagnostic follow-up changed only documentation; earlier results follow.

Follow-up full suite: `./test.sh` → `5 failed, 95 passed in 1.86s`.
All five fail because existing callers omit the newly required
`TrainingConfig.logging_steps`; interval logging passes. Next: user defaults
the config interval to `None`, aligns Trainer's default for direct callers, and
reruns the suite. This follow-up changed only documentation.

Latest check: at the user's explicit implementation request, Trainer now emits
interval mean loss and flushes the final partial interval. Logging plus Trainer
loss/empty/evaluation tests pass: `12 passed in 1.54s`; full suite not run.
Config and Builder wiring now reach the loop. Existing defaults still need to
restore agreed omitted-setting compatibility (TrainingConfig requires the
interval; Trainer defaults to 10). Next: resolve those defaults and cover exact
interval completion/disabled logging. The initial Red recorded below is historical.

On branch `feat/Trainer-Log`, HEAD `6476e04` (minimum evaluation committed), the
user selected `logging_steps` on `TrainingConfig`, wired through Builder. First
behavior: emit the mean of step losses every N completed optimizer updates and
flush the final partial interval; omitted setting preserves disabled automatic
logging. First integration test is Red: `TrainingConfig` rejects `logging_steps`.
Latest `./test.sh` on 2026-11-13 → `1 failed, 99 passed in 1.83s`; baseline before
the test was `99 passed in 2.07s`. Next: user adds/configures the interval and
implements loop emission, then verify Green. Further metrics, cumulative remote
steps, token-count policy, and scheduled eval are planned in 019, not yet tested.
The agent changed only tests/docs. The self-written eval follow-up below is
deferred while working on train-loop logging.

Previous direction (deferred): [017 — Self-Written Evaluation](017-small-evaluation.md).
On 2026-11-13 the user stopped [018's Flock evaluation integration](018-flock-final-evaluation.md)
and requested a self-written evaluator. Reuse `Trainer.evaluate`; the scope
question is pending: FineWeb-Edu dev loss/PPL first, or all three benchmark
families. No new evaluation API/tests or dependencies are introduced until the
scope is clear. Flock logging in task 013 is separate. Do not delete the user's
Flock checkout or continue its setup. Most recent full suite: `99 passed in
1.96s` (before this scope change); current targeted result is recorded in 017.
Next: settle custom evaluation scope and data preprocessing, then identify the
smallest missing behavior for the next TDD step. The supplied lab requirements
still specify Flock for official scoring; acceptance of a substitute is unconfirmed.

Previous task: [017 — Small Development Evaluation](017-small-evaluation.md).
The user accepted a small local dev evaluation and final scoring through Flock.
Agreed first API: `Trainer.evaluate(batches)` returns `{"loss": float,
"num_targets": int}` using target-weighted mean loss for dense, unpadded targets
and a mean-reduced supplied loss; the LM caller derives perplexity.
On branch `feat/Trainer-Log` at `1d2dc74` (PR #6 merged), the user implemented
`evaluate` and moved mode restoration to `finally`. Latest checks on 2026-11-13:
targeted eval/training/LM command → `11 passed in 1.67s`; full `./test.sh` →
`99 passed in 1.38s`. Weighted loss/types, eval mode/no-grad, normal/loss-error
mode restoration, empty list/iterator rejection, and preservation of parameters,
existing gradients, populated optimizer state, and training progress are Green.
The tiny LM now uses public `evaluate` before/after real training: 64 targets,
initial loss above 1, final loss below 0.05. New cases passed on first run;
no artificial Red is claimed. The agent changed only tests/docs.
Next: prepare a fixed real dev subset and agree on its preprocessing, then verify
manual evaluation with a real causal LM before agreeing on automatic scheduling.
Flock execution, GPU/masking/distributed behavior, and real dev metrics remain
unverified. Detailed commands and historical failures are in 017.
The older branch/next-action records below are historical context.

Flock eval documentation was successfully read on 2026-11-12 at documentation
revision `710e75a857969be3fefef67c50b3df016f15ef15`; details and links are in 017.
Important: official `fineweb_edu` scores documents separately with 8192-token
windows and rejects a shorter model context. A local 4096 packed eval remains a
development diagnostic. Limited in-memory scoring is available through
`prepare_task(..., limit=N)` and `score_prepared`. No Flock execution, package
installation, or model export compatibility has been verified. Confirm the
planned model can support 8192 before a long training run.

Read [Architecture.md](../docs/Architecture.md), [Trainer.md](../docs/Trainer.md), and [013 — Flock Logger and Delivery Verification](013-flock-logger.md), then inspect the relevant source and tests. Former tasks 013 and 014 are merged into task 013. The agreed public class is `upwash.core.logger.FlockLogger`, implemented by the user in `core/logger/flock.py` and re-exported by `core/logger/__init__.py`. The test is `tests/unit/core/test_flock_logger.py`. The user writes production behavior; the agent supports design, tests, failure analysis, and documentation under [AGENTS.md](../AGENTS.md).

## Current Position

As of 2026-11-12, PRs #1–#6 are merged. `main` is at `1d2dc74` (PR #6: tasks 015 and 016).

Configuration area, current task: [017 — Optimizer and schedule configuration for the sweep](017-optimizer-schedule-config.md), on branch `feat/optimizer-schedule-config`. Agreed interface: `OptimizerConfig(lr, muon_beta, adam_beta1, adam_beta2, adam_lr_multiplier, weight_decay)` and `ScheduleConfig(warmup_ratio, decay_ratio, decay_shape, decay_power)`. Every field is required. The builder passes only `lr` to the optimizer factory, and building Muon plus AdamW is task 020. Red verified on 2026-11-12: `47 failed` for unexpected new fields and `ImportError` for `ScheduleConfig`. The user implemented both as dataclasses, plus the builder change. Green was reached the same day: `78 passed` for the task, full suite `160 passed`. Committed as `7a8ed2c`.

Next configuration task, in the same pull request: [018 — Data configuration and `build_dataset`](018-data-config.md). Agreed interface: `DataConfig(root, pattern, format, seq_len, num_tokens)` and `build_dataset(config)` in `builder.py`. Red verified on 2026-11-13: `ImportError` for both names, while the rest of the suite gives `160 passed`. The user implemented both, and Green was reached the same day: `28 passed`, full suite `188 passed`. A real Flock folder builds 7,324 samples for a 30M-token budget. Committed as `2659838`.

Third configuration task, in the same pull request: [019 — Model configuration](019-model-config.md). Agreed interface: `ModelConfig(vocab_size, n_layers, d_model, d_ff, n_heads, n_kv_heads, max_seq_len, rope_theta, norm_eps, embedding_std, output_std_coeff)` with `head_dim` and `num_parameters()`. Tied embeddings use option B (embedding std 0.1, output multiplier `α = (coeff/d)/embedding_std`). `num_parameters()` is tested against all five spec reference rows. The 150M row's 36 KV heads is a typo for 3. Red verified on 2026-11-13: `ImportError`, while the rest of the suite gives `188 passed`. The user implemented it, and Green was reached the same day: `57 passed`, full suite `245 passed`. Next: commit, then open the pull request for 017–019. Task 020 builds Muon plus AdamW, and `TrainingConfig` with the experiment config follows in a later pull request. A teammate is working on logging inside the training loop.

Configuration (010), MemoryLogger (011), Builder (012), and the three FlockLogger unit cases are now verified Green. The user added deep-copying before SDK forwarding; all 65 tests pass. Task 013 remains incomplete because actual Flock delivery has not been exercised: current tests substitute RecordingRun for the SDK run.

The accepted constructor is `FlockLogger(project, name, config=None)` and initializes the SDK run from those settings; `log(metrics, *, step)` forwards events. The caller supplies experiment settings and owns logging policy, Flock host configuration, and authentication. The user implemented finish() and an atexit fallback; repeated-finish and post-finish behavior remain untested. Builder passes through the optional logger and Trainer stores it without automatic metric emission. The official Flock guide confirms use of the wandb SDK; W&B service support is outside scope. No SDK serialization or Flock delivery has been verified. Only finishing the unit test does not complete task 013.

```python
logger = FlockLogger(project="lab6-predictable-scaling", name="tiny-training", config={"lr": 0.1})
```

The earlier `FlockLogger(run=run)` constructor is superseded. Unit tests replace `wandb.init` with a RecordingRun factory; they do not initialize a real SDK run.

The required Flock scenario logs these seven metrics: `train/loss`, `train/grad_norm`, `train/learning_rate`, `train/tokens_per_second`, `train/total_tokens_seen`, `eval/loss`, and `eval/perplexity`. Their definitions and known-value payload are recorded in [013](013-flock-logger.md). Unit tests verify forwarding of supplied values; their numerical calculation remains with training/evaluation code. No automatic Trainer metric emission or logger key filtering is added. Parameter-group LR naming remains to be agreed for a future multi-group case.

Data area: tasks [015](015-multi-shard-dataset.md) (`ConcatDataset`, `Truncate`) and [016](016-npy-token-shard.md) (`PackedTokenDataset.from_npy` for Flock `.npy` shards) merged through PR #6. Still open for Flock data: verifying dev-set exclusion. Next for data: seeded shuffle, per-rank sharding, and resume (a sampler), then the data settings in task 018.

## Current Verification — 2026-11-12

Latest after the user's mapping-copy implementation: `./test.sh` → exit 0; `65 passed in 1.95s`. Constructor, seven-metric forwarding (values/step/commit/input isolation), and exit-callback regression are Green. This supersedes the earlier Red records below. All Flock tests patch wandb.init and use RecordingRun; no real SDK run or upload occurs, even with project lab6-predictable-scaling. Source search found no separate real Flock logging scenario under src/scripts/tests. The requested dashboard URL could not be inspected with the web tool, so no claim is made about existing server runs. Next: prepare actual Flock delivery verification with the selected organization/project, host/key setup, a distinct run name, and explicit completion; obtain authorization for the concrete hosted upload before execution. No credentials were read or changed and no network logging occurred.

Latest after explicit commit forwarding: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.23s`. The metric-event test now verifies all seven metric values, step 7, commit=True, and unchanged caller input immediately after log. Its final assertion fails: clearing the caller's metrics also clears RecordingRun's received mapping (`{}` instead of seven values). This confirms the intended mapping-isolation Red. Next: user forwards a separate dict to the SDK, preserving the current step/commit behavior. The full suite was not rerun. Earlier commit/export failures below are historical. The user also added an optional commit argument; commit=False behavior has not been agreed or tested.

Latest after the user's package export: `./test.sh tests/unit/core/test_flock_logger.py` → exit 1; `1 failed, 2 passed in 0.25s`. Constructor and exit callback pass. Metric forwarding now reaches `flock.py:20` and raises `TypeError: RecordingRun.log() missing 1 required keyword-only argument: 'commit'` because production calls `run.log(metrics, step=step)`. This is the intended forwarding Red for the agreed explicit commit behavior. Input-isolation assertions have not executed; the source still passes the caller's mapping directly. Next: user forwards an independent mapping and `commit=True`, then rerun the targeted tests. The full suite was not rerun for this check. Earlier missing-export outcomes below are historical.

Latest cleanup verification: `./test.sh tests/unit/core/test_flock_logger.py` → `1 failed, 2 passed in 0.24s`; `./test.sh` → `1 failed, 64 passed in 1.80s` (both exit 1). The sole failure is the missing FlockLogger package export. Constructor and captured exit-callback cases pass; no ignored atexit exception appeared. The error was a missing finish() method on the RecordingRun test double, reproduced Red and then fixed in test support. Tests intercept atexit registration to prevent process-exit callbacks leaking across cases. Production code was unchanged; see [013](013-flock-logger.md) for commands and the regression cause.

The earlier table below predates the cleanup regression:

| Command | Result | Interpretation |
| --- | --- | --- |
| `./test.sh tests/unit/core/test_flock_logger.py tests/unit/core/test_logger.py` | Exit 1; `1 failed, 5 passed in 0.28s`. | Required-project constructor and four MemoryLogger cases pass. Metric test is Red for the missing package export; seven-key payload assertions have not executed. |
| `./test.sh tests/unit/core/test_flock_logger.py` (earlier snapshot) | Exit 1; `2 failed in 0.02s`. | Historical syntax blocker, now resolved. |
| `./test.sh` (earlier project/payload) | Exit 1; `1 failed, 63 passed in 2.04s`. | Earlier regression evidence only; full suite not rerun after updating the fixture. |

Tests now use project `lab6-predictable-scaling` and the exact seven metric names from the user's requirements. Both logger cases patch SDK initialization and create no real run. The agent changed only tests and documentation; dependencies, source, credentials, and SDK settings were not changed.

Next: user exports FlockLogger through `core/logger/__init__.py` and completes metric forwarding, then rerun the targeted tests. Use explicit finish() when verifying delivery with the same project and metrics; its broader lifecycle semantics still need tests. Default/empty-config behavior, metric computation, real SDK serialization, and actual Flock delivery remain unverified. SDK is 0.31.0; the server/environment and exact deployed version remain to be selected.

## Route 011–013

| Task | Deliverable | Dependencies | Next action |
| --- | --- | --- | --- |
| [011](011-metric-logger.md) | Shared logging interaction and MemoryLogger. | 010 complete. | Complete for its four tested memory behaviors. |
| [012](012-build-trainer.md) | Build configured optimizer and Trainer with optional logger storage. | 010 complete. | Complete; existing Builder tests remain Green. |
| [013](013-flock-logger.md) | FlockLogger(project, name, config=None) and actual Flock delivery. | 011; scheduled after 012. | Constructor and exit callback Green; All unit cases Green; next verify actual Flock delivery. |

Former task 014 is merged into 013 and is no longer a separate milestone. Earlier tasks keep their numbers.

## Shared Design

- FlockLogger initializes its SDK run from `project`, `name`, and optional `config`, then exposes `log(metrics, *, step)` and forwards an independent metric mapping with the supplied step and `commit=True`. Only MemoryLogger exposes `records`.
- The caller supplies experiment settings and detached scalar metrics, and chooses logging frequency and valid increasing SDK steps. FlockLogger initializes the run; the user has implemented finish() with an atexit fallback; only the single callback path is tested. Returned `state.global_step` remains per-call; no cumulative Trainer logging counter is planned.
- Builder passes an optional logger unchanged to Trainer. Ordinary core/MemoryLogger imports must not require loading the remote SDK.
- A fresh Flock run created through the logger is the first delivery scenario. Resume, shared runs, CompositeLogger, multi-host delivery, and retry policies remain deferred.

Official guide links, SDK/documentation versions, historical adapter results, and delivery acceptance steps live in [013](013-flock-logger.md). Only an actual server check establishes delivery; offline SDK serialization does not.

## Historical Verification — Earlier on 2026-11-12

These entries predate the FlockLogger rename and merged task. Their old names, paths, and next-action notes are historical; the current verification above supersedes them.


- Latest full suite: `./test.sh` → exit 1; `1 failed, 62 passed in 2.28s`. Sole intended Red: `ImportError` for the missing `WandbLogger` in `upwash.core.logger`; metric assertions have not executed. All existing tests, including the data suite, pass. Earlier environment-blocker results below are historical.
- Local packages now include wandb 0.31.0, huggingface_hub 2.4.0, pyarrow 26.0.1, and pytest 9.2.0. The user's `pyproject.toml` declares `wandb>=0.31.0`; the agent made no dependency, lockfile, installation, or test-configuration changes. The user has since renamed the source to `flock.py`; it imports wandb but contains no logger class.
- Documentation read: the user-supplied hosted guide and its logging/compatibility references were successfully retrieved (HTTP 200), resolving the previous document-access blocker. The documented SDK transport is confirmed; deployed compatibility remains unverified. The pages identify documentation revision `710e75a857969be3fefef67c50b3df016f15ef15`. No tests were rerun and no source/test changes were made for this documentation-only update.
- Current branch is `main`, with PR #3 merged at `7e60792`. The worktree was clean before this session's documentation updates.
- `./test.sh tests/unit/test_training_config.py tests/unit/core/test_logger.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → exit 0; `14 passed in 2.95s`. Tasks 010–012 remain Green for their tested scope.
- `./test.sh` → exit 2; eight collection errors because the local `.venv` lacks `huggingface_hub` and `pyarrow`. This is an environment blocker, not an intended Red or a verified full-suite regression. No dependencies or test configuration were changed.
- Historical first Red at the superseded `integrations/` path: `./test.sh tests/unit/integrations/test_wandb_logger.py` → exit 1; `1 failed in 0.01s` for missing `upwash.integrations`. The user subsequently agreed to place WandbLogger in `core/logger.py`, and the test moved to `tests/unit/core/test_wandb_logger.py`. It expects one complete metric event with step 7, `commit=True`, and input isolation.
- Rechecked the same 010–012 command above after adding the test → exit 0; `14 passed in 1.29s`.
- After agreeing on `core/logger.py`: `./test.sh tests/unit/core/test_wandb_logger.py tests/unit/core/test_logger.py tests/unit/test_training_config.py tests/unit/test_builder.py tests/integration/test_built_trainer.py` → exit 1; `1 failed, 14 passed in 1.45s`. Intended Red: cannot import the missing `WandbLogger` class from the existing `upwash.core.logger` module. Behavioral assertions have not executed; existing regression cases remain Green.
- After the user-requested conversion to `core/logger/`, the same command → exit 1; `1 failed, 14 passed in 1.31s`. MemoryLogger's public import and existing behavior remain Green, and its moved source is byte-for-byte unchanged. WandbLogger remains Red for its missing class/export in the package. No SDK or dependency changes were made.
- Next: user implements and exports the run adapter in `flock.py`, then verify Green and select the Flock server/environment for 014. The last agreed public class is WandbLogger; align any rename before editing tests. An offline SDK check is an optional diagnostic, not a separate W&B milestone. Official documentation confirms the selected transport; actual delivery still requires execution against Flock.

## Historical Verification — 2026-11-09


| Command | Result | Interpretation |
| --- | --- | --- |
| `./test.sh tests/unit/test_training_config.py` | Exit 0; `6 passed in 0.01s`. | Current Config tests Green: composition, positive/zero values, negative rejection, NaN and both infinities. Broader input-type behavior is deferred. |
| `./test.sh tests/unit/test_builder.py tests/integration/test_built_trainer.py` | Exit 0; `4 passed in 2.05s`. | Construction, logger omission, real SGD updates at lr 0.1/0.05, and no automatic logging verified. |
| `./test.sh` | Exit 0; `46 passed in 2.11s`. | Current combined suite Green. |

Earlier environment and optional-logger failures are resolved. The two new integration cases passed on their first run; no Red is claimed for already-supported behavior. Detailed historical results remain in task files. The agent changed no dependencies or test configuration.

## Data Pipeline Handoff

Tasks [007](007-load-dataset.md) and [008](008-write-token-shard.md) retain their existing scope and records. Task 008 includes dev-id loading, resumable per-file shard generation, and `scripts/prepare_fineweb.py`. A recorded real-file run produced 726,000 documents and 731,876,929 tokens with matching shard-size and EOS-count checks. Its review updates record one snapshot request, shared parquet reading, and explicit little-endian `"<u4"` tokens, with historical Docker full-suite result `32 passed`.

Task 008 merged into `main` through PR #2. The skip-on-rerun check of the script has not been reported to the agent yet.

Task [009](009-packed-token-dataset.md) merged into `main` through PR #3. `PackedTokenDataset(path, seq_len)` memory-maps a `"<u4"` shard and returns `int64` `(x, y)` next-token pairs for `DataLoader`. A tiny LM trained through `DataLoader` and `Trainer` reaches loss below `0.05`. The real 008 shard with `seq_len=4096` yields 178,680 samples and `(8, 4096)` batches. On 2026-11-09, `main` (with tasks 010–012) was merged into the branch. The only conflict was this README. Full suite (Docker): `54 passed`. PR #3 review then asked for argument and index checks. `seq_len <= 0` now raises `ValueError`, negative indices count from the last sample, and out-of-range indices raise `IndexError`. Green: `15 passed` in the 009 unit file, full suite `62 passed`. PR #3 merged without the two suggested refinements. Both are still open and optional: check `seq_len` before opening the shard, and report the caller's index in the `IndexError` message. Only one real shard exists locally (`sample/10BT/000_00000.bin`, about 0.73B tokens).

## Gaps Toward the Lab Run

The lab target is a 1B Llama-3.2-style model trained on 200B FineWeb-Edu tokens with context 4096 and evaluated by Flock. This list is the agent's gap analysis against `main`. It has not been agreed, ordered, or assigned.

- Data: a seeded shuffle (reading many shards and a token budget are done in task 015), a resume position, per-rank sharding, and tokenizing about 200B tokens. One file took about 31 minutes for 0.73B tokens, so a single process needs on the order of 140 hours. Task 016 instead reads Flock's pre-tokenized `.npy` shards, synced to local disk.
- Model: no Llama implementation yet (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings), and no decision between writing one and depending on one.
- Training loop: no per-step loss output, learning-rate schedule (warmup and decay), gradient clipping, gradient accumulation, weight-decay parameter groups, bf16 mixed precision, device placement, checkpoint save and resume, multi-GPU training, or validation loss.
- Logging: FlockLogger unit cases are Green (013, PR #5); actual delivery to Flock is unverified. Useful metrics depend on `train()` exposing a per-step loss.
- Evaluation hand-off: the checkpoint format Flock evaluation expects has not been confirmed.
- Optional: μP (per-group learning rates, output multiplier, coordinate check). It covers width only, and the lab has not said whether it is required.

Tasks 001–008 keep their numbers, and 009 is the packed-token dataset. Configuration and logging originally used 010–014; former task 014 is now merged into 013.

## Completed Route

| Task | Behavior | Evidence | Status |
| --- | --- | --- | --- |
| [001](001-kernel-loss.md) | Compute scalar loss using supplied MSE and L1 loss functions. | Expected values `4` and `2`; tests now supply the required optimizer. | Green. Original API description is historical. |
| [002](002-minimal-training-loop.md) | Run a minimal training loop. | Real updates, fresh gradients, training mode, and per-call progress across successive calls. | Green, including repeated-call progress; supersedes former tasks 002–005. |
| [003](003-model-forward-hooks.md) | Use model output modified by a forward hook. | Hook-modified loss `1`; full suite `6 passed`. | Green. |
| [004](004-empty-batches.md) | Reject iterables that yield no batches. | Empty list and iterator raise `ValueError`; full suite `8 passed`. | Green. |
| [005](005-data-loader.md) | Minimal dataset and loader consumed by `Trainer.train`. | Ordered batches `[2, 2, 1]`, `drop_last`, `len` including divisible cases, repeated iteration, trained weight `0.72`. | Green: full suite `18 passed`. |
| [006](006-fit-linear-regression.md) | Fit `y = 3x + 1` end to end with the existing framework. | Final loss below `1e-4`; weight and bias within `1e-3` of `3` and `1`. | Verified: `19 passed`; no production change needed. |
| [007](007-load-dataset.md) | Read selected columns from pinned parquet files on the Hugging Face Hub, reusing the local cache. | Matching files in sorted order, requested columns, `ValueError` when nothing matches, pinned snapshot request. | Green: `23 passed`. |
| [008](008-write-token-shard.md) | Exclude dev ids, tokenize, append EOS, write a `uint32` token shard. | Exact token sequence, counts, no partial file on failure. | Green (`26 passed`); manual real-data check: 1,000 documents, 1,037,837 tokens, exact decode round trip. Wiring: `31 passed`; later snapshot-request and explicit little-endian review changes: `32 passed` in Docker. |
| [009](009-packed-token-dataset.md) | Read a `"<u4"` shard as packed `(x, y)` next-token samples for `DataLoader`. | Exact samples and shared boundary token, `len` cases, `int64` with large ids, tiny LM loss below `0.05`; `ValueError` for `seq_len <= 0`, negative indices, `IndexError` out of range, iteration stops at `len`. | Green: `40 passed` on the branch; `54 passed` after merging `main`; `62 passed` with review checks. Real shard yields 178,680 samples of 4,096 tokens. |
| [010](010-training-config.md) | Composed optimizer settings with finite nonnegative learning rate. | Six config cases pass locally on 2026-11-09; prior branch-local full suite `25 passed`. | Green for tested Python-float scope. |
| [011](011-metric-logger.md) | Batch scalar metrics in memory, ordered independent records, input isolation. | Four logger tests; full suite `42 passed` on 2026-11-09. | Green for tested memory scope. |
| [012](012-build-trainer.md) | Build configured optimizer and Trainer with optional logger storage. | Two unit cases and two real-SGD integration cases; full suite `46 passed`. | Green; no automatic metric emission. |
| [015](015-multi-shard-dataset.md) | Read many shards as one dataset and keep the first `num_samples` samples. | Ordered index mapping, empty parts skipped, negative and out-of-range indices, no crossing between real shards, budget checks. | Green: `21 passed`, full suite `83 passed`. |
| [016](016-npy-token-shard.md) | Read Flock's `int32` `.npy` shards with `PackedTokenDataset.from_npy`. | Same samples as `.bin`, `int64` with large ids, `seq_len` and index checks, two `.npy` shards in `ConcatDataset`; real Flock file memory-mapped. | Green: `21 passed` in the 009 file, full suite `89 passed`. |
| [017](017-optimizer-schedule-config.md) | Required Muon/AdamW optimizer fields and a WSD schedule config with validation; builder passes only `lr`. | Every field exposed and required, range and NaN/infinity checks, ratio sum, decay shapes, `to_dict`; builder tests with full configs. | Green: `78 passed`, full suite `160 passed`. |
| [018](018-data-config.md) | `DataConfig(root, pattern, format, seq_len, num_tokens)` and `build_dataset` (sorted glob → shards → `ConcatDataset` → `Truncate`). | Required fields, format and budget checks, sorted `.npy`/`.bin` reading, `offsets.npy` ignored, budget truncation and over-budget/no-match errors, `DataLoader` batches. | Green: `28 passed`, full suite `188 passed`. |
| [019](019-model-config.md) | `ModelConfig` for Llama-3.2-style models with tied-embedding init settings, `head_dim`, and `num_parameters()`. | Parameter counts equal all five spec reference rows (150M with 3 KV heads), required fields, positive sizes, head divisibility, even head dim, finite positive scalars, `to_dict`. | Green: `57 passed`, full suite `245 passed`. |

Detailed implementation iterations, test commands, and failures remain in each task document. Returned-state snapshot semantics and partial-failure recovery remain deferred. Custom operation kernels are not required for this milestone.

## Maintaining This Record

Keep the current task, observed results, blockers, and next action here. Keep detailed evidence in task files. Label API proposals, implemented behavior, historical results, and current verification separately. Do not count an unrun test as Green or an environment failure as the intended Red. Update the roadmap after each meaningful TDD step.
