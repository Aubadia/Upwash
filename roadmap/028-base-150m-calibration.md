# 028 — 150M base calibration with inverse-depth residuals, FA3 and compile

## Completion verified — 2026-11-15

Read-only recheck during task 030: job 778934 is COMPLETED (0:0), elapsed
03:23:04. Saved result.json confirms 228,881 updates / 29,999,890,432 tokens,
12,132.3046 seconds training+eval, and HF export relative error .00569855.
The Slurm log summarizes final eval/loss **2.78278** and train/loss 2.97692.
This supersedes the startup status below. No new full-history stability audit
was performed. The subsequent 300M sweep is a configuration proposal only;
see [030](030-300m-sweep-config.md). User requested no submission.


## Half-LR training authorized and launched — 2026-11-15 02:28 (Asia/Taipei)

The user explicitly approved starting the proposed half-LR 30B experiment.
Submitted **778934** to `8gpus`, eight GPUs / 48 CPUs / 14 h; startup check
shows RUNNING on **25a-hgpn052**. Configuration:
`configs/base150m/base150m-30b-seed2-lr0.002-am1.6.json`. Relative to the old
1.6 configuration, only base LR (.004 -> **.002**) and the run name change;
Adam multiplier remains 1.6, making AdamW LR **.0032**. Seed 2, data order,
global batch 32, WSD warmup .001 / decay .1, clip 1, and the exact
29,999,890,432-token / 228,881-update budget are preserved.

Uses task 029's accepted default-mode speed recipe: native compiled objective,
fused AdamW, prefetch, plateau-only compiled optimizer. Those implementation
changes passed short numerical/integration checks but are not bitwise identical
to the historical runs. The experiment tests stability with the accepted fast
runner; it does not isolate LR from all historical implementation differences.
No intermediate checkpoint/resume feature was added; final HF export remains
the existing behavior. Long-run success is not established by startup.

Prelaunch checks: programmatic config comparison confirms only LR/name changes
and exact token accounting. `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh
tests/integration/test_speed_objective.py tests/unit/test_run_config.py --tb=short
--show-capture=no` -> **9 passed in 9.15s**.

Frozen source/config/plan and SHA256 manifest:
`/work/asciibase64/upwash/snapshots/base150m-half-lr-fast-20261114T182744Z`.
Submission record: `/work/asciibase64/upwash/base150m-half-lr-pipeline.json`.
Slurm log: `/work/asciibase64/upwash/logs/base150m-half-lr-778934.out`.
Run name: `fast-base150m-30b-seed2-lr0.002-am1.6-778934`.
Startup verified: at step **400**, 52,428,800 tokens, train loss **6.378932**,
pre-clip norm **.191777**, Muon LR **.002**, AdamW LR **.0032**, interval
throughput **2,460,805 tokens/s**. Logged losses decreased from 11.405538 at
step 100 to 7.354013 at step 300; inspected values are finite. These are early
training metrics, not evidence of late-run stability. Flock history uploads
returned **200 OK** (history offsets 0, 2, 4). Online run:
https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/oby75muf
Next: assess full-dev metrics and stability beyond both historical onset points
(~15.57B / ~20.77B) through the full budget. Training remains running.

## Follow-up diagnosis verification — 2026-11-15

Re-read the retained local binary histories with
`experiments/sweep150m/read_wandb_history.py` and checked `sacct` for jobs
777067/777068. Both remain CANCELLED. Pipeline metadata maps the user-named
`base150m-30b-seed2-am1.6-777055` to formal job 777067 / run 9l1squnr;
777055 is the validation launcher suffix, not this formal job ID.

- The recorded onset and dev losses below are confirmed. Final available
  training records are step 170300 / 22,321,561,600 tokens / loss 7.97324 /
  pre-clip norm 345120768 for multiplier 1.6, and step 168900 /
  22,138,060,800 tokens / loss 6.35320 / pre-clip norm 8939.73 for 1.5.
  These replace earlier partial-history endpoints, not the onset estimates.
- The 1.6 output directory contains provenance/config files but no saved model
  checkpoint. Current source has no intermediate checkpoint/resume path.
- Current-worktree CPU regression command:
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/unit/core/test_trainer_clipping_and_lr.py tests/unit/components/optimizers/test_muon_adamw.py tests/integration/test_muon_adamw_trainer.py`
  -> exit 0, **9 passed in 29.01s**. This confirms existing tested behavior;
  it neither reproduces the long-run failure nor establishes its root cause.

Recommendation for discussion, not an agreed implementation or submitted run:
first add resumable model/optimizer/scheduler/RNG/data-position checkpoints and
per-group update/parameter norms, pre/post-clip norms, and interval maxima.
Then test a lower base LR (initial candidate .002 with multiplier 1.6 held
fixed), keeping the 30B schedule and other settings constant. Passing a short
3B run is insufficient; verify beyond both historical onset points and through
the full budget. A replay from a pre-onset checkpoint can later separate Muon
and AdamW LR effects. Lower LR is a stability hypothesis, not a proven fix.
No production code, configs, dependencies, or job states changed in this check.
Next: agree on the diagnostic/checkpoint behavior and controlled LR experiment.

## User-requested cancellation — 2026-11-15 00:18 (Asia/Taipei)

The user explicitly requested stopping both degraded runs. Executed
`scancel 777067 777068`; exit 0. Slurm accounting confirms both CANCELLED
(elapsed 03:12:25 and 03:11:34 respectively); the first queue check showed
COMPLETING while resources were being released. Existing logs, local Flock
histories, run configs and source snapshots are retained. Neither run completed
the 30B budget. No replacement run was submitted. The instability diagnosis
below remains unresolved; a replacement calibration requires an agreed plan.

## Loss instability investigation — 2026-11-15 00:15 (Asia/Taipei)

Both formal jobs remain RUNNING. Local Flock/W&B binary histories confirm
sustained degradation, not just a dashboard artifact. These observations
supersede the earlier statement that long-run stability is merely untested.

- Multiplier 1.6 / job 777067 / run 9l1squnr: at step 118700 train loss
  3.07779 and pre-clip gradient norm .5564; at 118800 (~15.571B tokens)
  loss 3.18999 and norm 10.0865; at 119100 loss 4.04808 and norm 1494.78.
  Dev loss rose from 2.86903 at 114440 to 6.39225 at 137328 and 7.88503
  at 160216. Latest sampled step 167600: train loss 7.94456, ~21.968B tokens.
- Multiplier 1.5 / job 777068 / run 8w9460qn: degradation starts around
  158400–158600 (~20.762–20.788B tokens). Dev loss rose from 2.85684 at
  137328 to 7.43730 at 160216. Latest sampled step 166200: train loss
  6.59515, ~21.784B tokens. Neither history contains nonfinite train loss
  in the inspected records.
- Learning rates remain .004 for Muon and .0064/.006 for AdamW throughout
  the onset. Decay has not started. Loss is the rank-zero 100-update mean;
  gradient norm is the final update's pre-clip global norm, not the maximum
  over that interval. Exact onset cannot be recovered from interval logs.
- Both actual run configs specify max_grad_norm=1; snapshot Builder passes
  this to Trainer, which clips after backward and before optimizer.step.
  Snapshot CPU checks:
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/unit/core/test_trainer_clipping_and_lr.py tests/unit/components/optimizers/test_muon_adamw.py tests/integration/test_muon_adamw_trainer.py`
  -> **9 passed in 18.29s**. This does not measure post-clip GPU gradients
  at the failure step; those and parameter/update norms are not logged.
- Installed torch.optim.Muon normalizes its momentum-derived update before
  Newton–Schulz iteration. A CPU first-step example (32x32 matrix, seed 2,
  lr .004, momentum .9, match_rms_adamw, no decay) changes gradient norm
  3145.539 to 1 with clipping, while update norm changes only .023594 to
  .023498. This explains why raw-gradient clipping does not proportionally
  bound Muon updates; it does not establish the trigger of this failure.
- Hypothesis: the short-budget HP winners are unstable over the longer
  constant-LR phase. Root cause remains unproven; no attribution to FA3,
  compile, residual scaling, or a particular parameter group is established.
  No intermediate model/optimizer checkpoints exist for replay. The 240-step
  smoke established execution correctness within its scope, not 30B stability.
- Logs also report Flock upload failures (502/404 and fatal upload warning).
  Local histories continue advancing; remote completeness is not established.
  No compiler traceback/recompile-limit failure was found in inspected logs.

No training settings or job states were changed during this investigation.
Next: decide whether to terminate the degraded runs; before a replacement
calibration, agree on stability diagnostics/checkpointing and a controlled
LR comparison. Do not treat either current run as a successful calibration.

## Latest verified status — 2026-11-14 21:05 (Asia/Taipei)

Validation **777055** completed successfully on H200 node 25a-hgpn124 using
snapshot v3. Full 8-GPU FA3 + compile smoke completed 240 updates / 31457280
tokens, both dev evaluations, export, and online Flock sync. No Inductor error,
traceback, or recompile-limit fallback occurred. Cache isolation plus the
scoped compiler-limit change resolved the observed smoke failures on this run;
it does not establish the exact underlying cause of the old cache failure.

- Final smoke dev loss: 5.6685294113; perplexity: 289.6083264.
- Last interval throughput: 1111631.43 tokens/s. Entire training script: 105 s,
  including startup, compile, evaluations, export and logging.
- Flock smoke: https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/45up3a2n
- Formal jobs automatically submitted through calibrate_base.sbatch:
  **777067** (Adam multiplier 1.6), **777068** (1.5). Latest check: 777067 RUNNING on 25a-hgpn025; 777068 PENDING under
  MaxGRESPerAccount. Each requests 8 GPUs and 14 h walltime. Same seed 2,
  residual 1/6, FA3/compile, 228881 updates / 29999890432 tokens per run.
- Pure training projection ~7.5 h/run from the smoke's last interval, excluding
  compile/evaluation/export; no final calibration results exist yet.
- All previous validation attempts below are historical. Long-run stability and
  final losses remain pending. Do not treat the smoke loss as a 30B result.

## Authorized scope (2026-11-14)

The user approved agent implementation on `experiment`, dependency preparation,
data under `/work/asciibase64`, and two independent 150M runs, seed 2, ~30B
training tokens each. All GPU execution must use `.sbatch`. Training must log
online to Flock project `lab6-predictable-scaling`. No 300M scale-up or decay
sweep is authorized by this task.

- 6 layers, width 768, FFN 3072, 12 query/3 KV heads, context 4096.
- Each attention and MLP residual branch is multiplied by 1/6.
- Muon API LR .004, beta .9, match_rms_adamw; Adam betas .9/.99.
- Compare Adam multipliers 1.6 and 1.5. Weight decay config .04, API .04/768.
- BF16, clip 1, 8 GPUs x batch 4; data shuffle seed 0.
- WSD: warmup .001, decay .1, power 1.0. 228881 updates/run,
  29999890432 tokens/run; 228 warmup updates. User accepted batch alignment.
- Keep existing initialization and output multiplier; fold residual and output
  multipliers into the exported HF model and verify reloaded logits.
- Best is lowest final full-dev loss, not the minimum intermediate evaluation.

## Current verification

CPU only (`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2`):
- `./test.sh tests/unit/test_adaptor.py tests/unit/test_export.py tests/unit/test_train.py tests/unit/test_run_config.py`:
  21 passed in 32.35s.
- `./test.sh tests/unit/test_residual_scaling.py`: intended Red, 1 failed in
  24.38s. Reference HF 5.22.2 forward/backward succeeds before Upwash fails
  because ModelConfig has no residual_scale_by_depth keyword.
- Post-implementation targeted regression: 78 passed, 1 failed because the old
  ModelConfig.to_dict test did not expect the newly added optional field. Updated
  that expectation to include residual_scale_by_depth=False.
- Full CPU regression after that adjustment: `./test.sh` -> 377 passed in 19.34s.
- Submission-gate tests: 6 passed (reject partial token counts, NaN, incomplete,
  offline, and zero-throughput smoke output).
- After adding source SHA256 to Flock metadata:
  `./test.sh tests/unit/test_train.py tests/unit/test_run_config.py tests/unit/test_calibration_submission.py`
  -> 20 passed in 26.08s. GPU checks remain unrun, not marked Green.
- `git diff --check` and shell syntax checks pass.

## Data and environment

The old `/work/u7725908` pool is not readable. Rebuild the pinned official HF
FineWeb-Edu source with existing dev-ID exclusion and pool seed 0, target 205B.
Preparation job 775793 was pending (MaxGRESPerAccount), cancelled before start,
and replaced by job 775799 with 4 GPUs/48 CPUs on dev (CPU preparation only).
Existing unrelated Mirage job is left untouched. Completed shards are reusable
if the preparation time limit is reached.

FA3 official source pinned to f63c472c4dc19237c0635d6a629df065ff64796e under
`/work/asciibase64/upwash/vendor/flash-attention`. Build via
`scripts/build_fa3.sbatch`, CUDA toolkit 13.0, targeting Hopper BF16 head dim 64
forward/backward. Torch 2.15.0 and Transformers 5.22.2 remain unchanged.
The pinned source includes the top-level flash_attn_interface compatibility
module used by Transformers. Flock and HF credentials exist; values are never
printed or written into this record.

## Active pipeline

- Data preparation: **775799**, running; 9 complete shards, 2351883716 tokens at the
  last check. The manifest is not complete yet.
- FA3 build: **775838**, queued (QOSMaxJobsPerUserLimit).
- 8-GPU validation: **775839**, queued with afterok dependencies on both.
- Earlier pending build 775803 and validation 775814 were cancelled before
  execution and replaced to run from the frozen source snapshot.
- Snapshot: `/work/asciibase64/upwash/snapshots/base150m-20261114-seed2`, with
  source-manifest.json file hashes. Its .venv and credential file are private
  symlinks to the existing environment; no credentials are copied into artifacts.
- Pipeline state: `/work/asciibase64/upwash/base150m-pipeline.json`.
- Flock login against flock-api.cerulean.works verified successfully; this does
  not yet establish GPU smoke delivery. Tokenizer fully cached for offline jobs.
- Dev preparation: 5000 docs, 5360078 tokens; scheduled data verification will
  check BOS/EOS and 1308 packed dev samples.
- Formal 30B jobs are **not submitted yet**. Validation runs GPU numerical
  comparisons and actual FA3 profiling, then a 240-update/31457280-token,
  8-GPU compiled smoke with online Flock. Only finite metrics, exact token
  count, successful export/script completion and an online run URL allow its
  launcher to submit the two formal .sbatch jobs.
- Requested formal walltime = ceil(1.5 * measured token time + 2 hours), capped
  by rejecting projections above 48 hours. The two jobs are independent and
  may serialize under the account GPU limit. No existing Mirage job is changed.
- Runtime logs: `/work/asciibase64/upwash/logs/`. Validated results and final
  comparison are saved under `/work/asciibase64/upwash/validation/`; each
  successful model directory gets result.json with the Flock run URL.
- Current limitation: GPU/FA3/compile integration and real online delivery await
  allocated resources. Any failed prerequisite stops subsequent submission.

## Next

Verify CPU Green, FA3 build/GPU numeric checks, data manifest and dev dataset,
then 8-GPU compiled online Flock smoke. Submit the two production jobs only
after these gates pass, with wall time based on measured throughput. Record
job IDs, Flock links, actual outcomes and any blocker here.

## Follow-up status and smoke failure — 2026-11-14 evening

- Preparation 775799 completed, exit 0, 03:12:01. Final manifest:
  326 shards, 205087998552 tokens, 203705949 documents. Data acceptance passed.
  Dev-ID exclusions are zero; dev parquet contains 5000 records from
  CC-MAIN-2025-26. Zero exclusions is recorded as observed, not proof of text
  deduplication across train/dev. The pinned ID exclusion rule remains active.
- FA3 build 775838 completed, exit 0, 00:01:55 on H200. Installed FA3 3.0.1;
  actual BF16 GQA forward/backward succeeded. Build metadata: fa3-build.json.
- Single-GPU comparisons and profiling passed. Compiled FA3 relative errors:
  logits .01010, gradients .01447, three-step update .05181, export .00996,
  within the predetermined BF16/NS tolerances. Profiler observed actual
  flash_attn_3 forward and backward operators.
- Validation 775839: online Flock run jcfq8wm9 opened and finished with zero
  history events; rank failure was obscured by collective cleanup. Cancelled
  this validation; added immediate exception printing before teardown.
  CPU `./test.sh tests/unit/test_train.py`: 7 passed in 28.48s.
- Diagnostic validation 777046 (snapshot v2) reached step 120 in Flock run
  vtfdga8r: 15728640 tokens, train loss 6.7850, dev loss 6.6304, ~1.027M tok/s.
  Another rank failed during evaluation compilation with InductorError /
  CUDA_ERROR_FILE_NOT_FOUND from cuModuleLoad. FA3 attention also exceeded
  Dynamo's eight-variant recompile limit and fell back to eager. Cancelled;
  no formal runs submitted. These smoke values are not calibration results.
- Fix under validation: separate Inductor/Triton/CUDA cache paths for each
  Slurm job and torchrun rank, with fresh numerical-validation caches;
  scoped recompile_limit=64 and fail_on_recompile_limit_hit=True.
  Shared/stale compiler cache paths are a suspected cause of missing cubins;
  the observed failure is established, the cause is not proven yet.
- Added submission regressions rejecting recompile fallback and Inductor errors.
  Red: 2 failed, 6 passed; after fixing the gate and runtime settings, targeted
  train + submission checks: 15 passed in 26.14s. Shell syntax/diff checks pass.
- A new fixed snapshot v3 and complete GPU validation have been submitted.
  Formal jobs remain gated on successful validation and online Flock output.
