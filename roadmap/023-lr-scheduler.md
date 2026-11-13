# 023 — Learning-Rate Scheduler Wiring

## Latest Green — Decay Shapes and Phase Boundaries

Expanded the existing WSD unit test to 13 explicit numerical cases, retaining
the agreed original 10-update example. New cases cover cosine quarter-turns,
power exponents 2 and 1.5, no warmup, no decay, stable-only, warmup-only,
decay-only, no stable phase, fractional counts truncated from 2.8 to 2,
sub-update counts truncated to zero, and a single decay update ending at zero.
Expected cosine values use square-root identities and power values use
independent fractions/radicals, rather than reproducing the scheduler formula.

Each case uses real CPU Muon/AdamW groups, checks the entire prepared LR
sequence against base LR times the expected factors (rel=1e-10, abs=1e-12),
checks exact final zero when decay is present, and verifies original child
optimizer dictionaries are updated. With no decay steps, including ratios
truncated to zero, no decay is applied; a warmup-only schedule ends at base LR.
All new cases passed immediately; no Red or production changes are claimed.

- `./test.sh tests/unit/components/schedulers/test_wsd.py tests/unit/test_schedule_config.py tests/integration/test_lr_scheduler.py tests/integration/test_muon_adamw_trainer.py --tb=short`
  → exit 0, **47 passed in 2.41s**.
- `./test.sh --tb=short` → exit 0, **296 passed in 3.38s**.

This increment changed tests and roadmap documentation only. No current
blocker. Next: agree on the next training increment, such as HF Llama output
adaptation. Invalid budgets/beyond-budget contracts, scheduler persistence,
no-logger/evaluation/cross-call scheduling guarantees and per-group LR metrics
remain separate work; GPU/bf16 is not covered by these CPU tests. The linear
alias remains outside ScheduleConfig's accepted values (use power exponent 1).
Earlier records below are historical.

## Latest Green — Real WSD Training Integration

`wsd.py` is implemented with linear/cosine/power branches, matching the agreed
linear schedule. This supersedes the older missing-module Red below. Existing
ScheduleConfig still accepts only cosine/power; linear is expressed as power
with exponent 1. The scheduler's additional linear alias does not change that
config validation contract.

Added a second test in `tests/integration/test_muon_adamw_trainer.py` for a full
10-update CPU float32 run through Builder, the real WSD factory, CombinedOptimizer
and Trainer. Independent real torch 2.15.0+cu132 Muon/AdamW references run first
on a deep-copied tied tiny LM, with explicit parameter groups and manually set
LRs from [0.5, 1, 1, 1, 1, 1, 1, 2/3, 1/3, 0]. Inputs are independent and RNG
state is preserved. No Upwash scheduler or optimizer is used by the reference.

Verified: both groups' actual pre-update LRs match their base LR times these
factors; the LR ratio is preserved; all model parameters change over training;
final parameters match the independent reference (rtol=1e-5, atol=1e-7); tied
identity is preserved; the tenth update uses exactly zero LR and leaves every
parameter exactly unchanged from its pre-update snapshot. MemoryLogger emits
steps 1–10 and the first group's used LR, matching current logging behavior.
Observations use the supplied batch iterator before each update, without
patching production code or asserting internal call counts. This does not
claim optimizer state is unchanged at zero LR.

- `./test.sh tests/integration/test_muon_adamw_trainer.py tests/unit/components/schedulers/test_wsd.py tests/integration/test_lr_scheduler.py --tb=short`
  → exit 0, **4 passed in 2.49s**.
- `./test.sh --tb=short` → exit 0, **284 passed in 3.64s**.
- The new test passed immediately against existing production code; no Red
  is claimed. This increment changed tests and roadmap documentation only.

No current blocker. Next: add independently calculated cosine/nonlinear-power
and zero-length/fractional-phase cases after agreeing on remaining boundaries.
Direct HF output adaptation and device/bf16 are later training milestones.
Per-group LR metrics, scheduler checkpoint state and invalid/beyond-budget
semantics remain separate work. Earlier records below are historical.

## Current Increment — Linear WSD Red

The user approved the rounding and endpoint conventions proposed below:
warmup/decay counts use int(total_steps * ratio), construction prepares the
first update, warmup begins at base_lr / warmup_steps, and the final decay
update uses exactly zero. Apply one common multiplier to each base LR.

Added `tests/unit/components/schedulers/test_wsd.py` with the agreed 10-update
example: two warmup, five stable and three linear-decay updates, using
`ScheduleConfig(0.2, 0.3, "power", 1.0)`. Real CPU Muon and AdamW optimizers
have base LRs 0.008 and 0.0128 and are wrapped by CombinedOptimizer. Both
construct successfully before importing the new scheduler. The test records
the LR prepared for each update and compares both groups to independently
specified factors [0.5, 1, 1, 1, 1, 1, 1, 2/3, 1/3, 0], using rel=1e-10 and
abs=1e-12. Final zero is checked exactly, including on each child optimizer's
original group dictionary. This is schedule-only unit coverage, without
backward/update calls; real Trainer timing is covered by the existing
integration test. No external scheduler compatibility oracle is selected.

- `./test.sh tests/unit/components/schedulers/test_wsd.py tests/unit/test_schedule_config.py tests/integration/test_lr_scheduler.py --tb=short`
  → exit 1, `1 failed, 32 passed in 2.52s`.
- `./test.sh --tb=short` → exit 1, `1 failed, 282 passed in 3.43s`.
- Sole failure: `ModuleNotFoundError: No module named 'upwash.components.schedulers'`
  at the agreed `build_wsd_scheduler` import. This is intended missing-API Red;
  numerical Upwash schedule assertions have not executed.

Next: user implements `build_wsd_scheduler(optimizer, *, config, total_steps)`
under `components/schedulers/wsd.py`, then verify Green. No production stubs,
dependencies or test configuration were added; changes are tests/docs only.
Fractional-step rounding, zero-length phases, nonlinear power/cosine, invalid
budgets, beyond-budget behavior and real WSD training still need subsequent
tests. Earlier records below are historical.

## Latest Green — Scheduler Advancement

The user now calls optional lr_scheduler.step() immediately after
optimizer.step(). The two-update timing test passes, including final weight
0.56, two scheduler steps, prepared LR 0.025 and logged used LRs [0.1, 0.05].

- `./test.sh tests/integration/test_lr_scheduler.py tests/unit/test_builder.py tests/integration/test_built_trainer.py --tb=short`
  → exit 0, `7 passed in 2.24s`.
- `./test.sh --tb=short` → exit 0, `282 passed in 3.40s`.

Next: agree on WSD numerical conventions before writing its first test.
Proposed API remains `build_wsd_scheduler(optimizer, *, config, total_steps)`
under `components/schedulers/wsd.py`. Proposed step counts are
`int(total_steps * warmup_ratio)` and `int(total_steps * decay_ratio)`.
The first warmup update uses base_lr / warmup_steps; the last decay update
uses zero. For 10 updates, warmup ratio 0.2, decay ratio 0.3 and power 1,
proposed used multipliers are [0.5, 1, 1, 1, 1, 1, 1, 2/3, 1/3, 0].
These rounding/endpoint choices are proposals awaiting user confirmation,
not implemented or tested behavior. No production/test changes in this
verification step. Earlier Red records below are historical.

## Current Increment — Wiring Green, Scheduler Advancement Red

The user completed Builder's schedule inputs and factory invocation. Trainer
retains the optional `lr_scheduler`. Baseline
`./test.sh tests/unit/test_builder.py --tb=short` returned exit 0,
`4 passed in 2.33s`: the scheduler wiring assertions now execute and pass.

Added `tests/integration/test_lr_scheduler.py` for the proposed update order.
The test injects a local halving scheduler via Builder, uses real CPU float32
SGD and mean MSE, and trains two batches with input 1, target 2, initial weight
0. Scheduler math is deliberately a test double, independent of WSD. Expected
learning rates used are 0.1 then 0.05: weight 0.4 then 0.56. It also requires
two scheduler steps, final prepared LR 0.025, and logged used LRs [0.1, 0.05].
This is an independently calculated integration test, not a differential test.

- `./test.sh tests/integration/test_lr_scheduler.py tests/unit/test_builder.py tests/integration/test_built_trainer.py --tb=short`
  → exit 1, `1 failed, 6 passed in 2.50s`.
- `./test.sh --tb=short` → exit 1, `1 failed, 281 passed in 3.32s`.
- Failing assertion: final weight expected 0.56, actual approximately 0.72
  (absolute difference 0.1600000262). Source calls optimizer.step() but never
  lr_scheduler.step(), so both updates use LR 0.1. This is the intended
  advancement Red; later scheduler-count/LR/log assertions have not executed.

Next: user calls `self.lr_scheduler.step()` after successful optimizer.step()
when a scheduler is supplied, preserving used-LR metric capture before the
update. Then rerun for Green. No production changes in this increment; only
test and roadmap additions. WSD math, evaluation inactivity, no-logger
scheduling, continuation across train calls and error behavior need later
increments. Earlier verification below is historical.

## Latest Recheck — Defaults Restored

The user added `None` defaults to Trainer's `lr_scheduler` and Builder's
`lr_scheduler_factory`; the 48 old constructor regressions are resolved.

- Targeted Builder/schedule-config/Muon integration command: exit 1,
  `1 failed, 35 passed in 2.33s`.
- `./test.sh --tb=short`: exit 1, `1 failed, 280 passed in 3.43s`.
- Sole failure: new wiring test gets unexpected keyword `schedule_config`.
  Builder still lacks `schedule_config` and `total_steps` inputs and calls
  `lr_scheduler_factory()` without arguments. Factory assertions have not run.

Next: user adds the two optional inputs and passes optimizer, config and
total_steps to the factory. No production/test changes in this check.
Earlier outcomes below are historical.

## Full-Suite Regression Check

At the user's report of 49 failures, `./test.sh --tb=short` reproduced
**49 failed, 232 passed in 3.67s** (exit 1). All 48 pre-existing failing cases
stop at construction because `Trainer.lr_scheduler` or Builder's
`lr_scheduler_factory` is required instead of optional. The new wiring test
stops because Builder rejects `schedule_config`. These are constructor/API
failures, not numerical optimizer or scheduler execution failures.

Next: user adds optional `None` defaults to the scheduler/factory inputs,
accepts `schedule_config` and `total_steps` in Builder, and forwards all three
factory arguments. Preserve existing callers; do not modify tests to supply a
mandatory scheduler. Trainer already stores `self.lr_scheduler`. This check
changed documentation only.

## Naming Correction

The user explicitly selected `lr_scheduler` for the Trainer argument and
attribute, alongside `lr_scheduler_factory` for Builder. Updated the wiring
test to assert `trainer.lr_scheduler`; this is an authorized API rename, not
a change to injection behavior. The source now stores `self.lr_scheduler`.
Both Trainer's `lr_scheduler` and Builder's factory must remain optional.
Earlier naming-discussion records below are historical.

After the rename, rerunning the same targeted command returned exit 1,
`5 failed, 31 passed in 4.24s`: Builder still rejects `schedule_config` and
requires `lr_scheduler_factory`. No full-suite run or production edits.

## Latest Implementation Check

The targeted command recorded below now returns exit 1,
**5 failed, 31 passed in 2.29s**. Builder accepts a required
`lr_scheduler_factory` but still rejects `schedule_config`; four existing
Builder/integration cases fail because the factory has no default. Source
inspection additionally shows that Builder calls the factory without optimizer,
config or total_steps, and Trainer requires `lr_scheduler` without saving it.
The prior agreed storage name/test expectation is `scheduler`; the user's
current `lr_scheduler` name has not been reconciled with that contract. No test
expectations were changed to match the incomplete implementation.

Next: make the new inputs optional, pass optimizer/config/total_steps to the
factory, and store its result on Trainer using the agreed name (or explicitly
agree on renaming before changing the test). Direct Trainer callers also need
the optional default. Scheduler stepping remains a later increment. No
production/test edits or full-suite run in this check; documentation only.

## Agreed Builder Interface

The user requested scheduler integration, chose to pass scheduler settings
directly to Builder instead of binding them with `partial`, and explicitly
named the injected factory `lr_scheduler_factory`.

```python
trainer = build_trainer(
    config=training_config,
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=optimizer_factory,
    lr_scheduler_factory=lr_scheduler_factory,
    schedule_config=schedule_config,
    total_steps=100,
)
```

Builder first constructs the optimizer using its existing contract, then calls
`lr_scheduler_factory(optimizer, config=schedule_config, total_steps=total_steps)`.
It passes that exact scheduler to Trainer via optional `lr_scheduler=None`, exposed
as `trainer.lr_scheduler`. Builder must not perform training or call scheduler
`step()`. New Builder arguments should be optional to preserve current callers;
omitting the factory preserves training without a scheduler. ScheduleConfig
remains standalone in this increment, without changing TrainingConfig.

## Verified First Red

Test: `tests/unit/test_builder.py::test_build_trainer_constructs_and_injects_configured_lr_scheduler`.
Uses a real CPU SGD through the existing test adapter and a local scheduler
stub. The factory checks config identity and total step budget; assertions
check that it receives Trainer's exact optimizer, Trainer retains its returned
scheduler and construction leaves model weights unchanged. The stub rejects
any premature `step()` call. No concrete scheduler implementation is required
for this wiring test, so this is not a differential scheduler test.

- Command: `./test.sh tests/unit/test_builder.py tests/unit/test_schedule_config.py tests/integration/test_muon_adamw_trainer.py --tb=short`.
- Result: exit 1, **1 failed, 35 passed in 4.24s**.
- Expected: configured scheduler factory is called and its result is injected.
- Observed: `TypeError: build_trainer() got an unexpected keyword argument 'lr_scheduler_factory'`.
- Cause: the new optional Builder inputs have not been implemented; factory
  body and scheduler identity assertions have not executed. Intended API Red,
  not an environment failure.
- Only tests and roadmap documentation changed. Full suite not rerun; last
  full-suite checkpoint before this increment was `280 passed in 3.42s`.

Next: user implements Builder construction/injection and optional Trainer
scheduler storage, then verify Green and add real update-order tests. The
proposed lifecycle is to prepare the first update's LR during scheduler
construction, call scheduler.step() after each successful optimizer.step(),
preserve scheduler progress across train() calls, and leave evaluation inert.
Training LR metrics should describe the update just performed. These lifecycle
behaviors and WSD endpoint/rounding conventions still need explicit tests and
agreement before their numerical increments.

WSD implementation is planned under `components/schedulers/wsd.py`, applying
the same multiplier to each group's base LR to preserve the Muon/AdamW LR
ratio. Local torch 2.15.0 LRScheduler source checks isinstance(optimizer,
Optimizer); the current CombinedOptimizer is not a subclass. Concrete scheduler
implementation strategy, total_steps validation, missing-setting errors,
checkpoint behavior and schedule mathematics remain subsequent work.
