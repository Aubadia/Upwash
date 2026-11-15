# 030 — 150M decay shapes and 300M scaled learning rates

## Formal jobs launched and startup verified — 2026-11-15 14:08 (Asia/Taipei)

Both full-model numerical comparisons and both 400-update / 52,428,800-token
eight-rank integrations pass the corrected acceptance checker. Each completes
four full-dev evaluations, all three group LRs reach zero, and HF export/reload
is within .03. Short-run results (not 60B convergence evidence):

| LR label | Final smoke dev loss | Train/eval seconds | Peak GiB | Export relative error |
| --- | ---: | ---: | ---: | ---: |
| .002 | 5.730574 | 151.990 | 19.382 | .002932 |
| .0025 | 5.456542 | 109.972 | 19.382 | .002958 |

Original job **780263** is FAILED (1:0) solely at the final checker KeyError;
all preceding GPU stages completed. The corrected checker passes the original
outputs. The original acceptance snapshot is retained. Launch snapshot
`/work/asciibase64/upwash/snapshots/group300m-20261115T055526Z-launch` differs
only in `check_acceptance.py`; all GPU Python source hashes were compared with
the accepted snapshot before launch. New hashes and submission.json are retained.

- **780284**, LR label .002: RUNNING on 25a-hgpn044, eight GPUs / 24 h.
  Output: `/work/asciibase64/upwash/runs/300m-60b-base-lr0.002-power-p1-780284`.
- **780285**, LR label .0025: RUNNING on 25a-hgpn064, eight GPUs / 24 h.
  Output: `/work/asciibase64/upwash/runs/300m-60b-base-lr0.0025-power-p1-780285`.

Both use 457,763 updates / 59,999,911,936 tokens, fixed power p=1, main/down/Adam
manifest LR/WD and the accepted default compiled-objective recipe. CPU suite
remains recorded as 390 passed before the checker regression was added; the
new checker plus existing LR-log contract separately gives 7 passed. No unrun
full-suite result is claimed.

Online startup is now verified from actual local Flock histories and HTTP 200
upload responses. Machine-readable pipeline records include each URL, observed
step and full latest metrics. All four jobs remain RUNNING.

| Job | Observed step / tokens | Train loss | Tokens/s | Flock run |
| --- | --- | ---: | ---: | --- |
| 780253, 150M p=1.25 | 16,500 / 2,162,688,000 | 3.27208 | 2,504,098 | qh7g8hkd |
| 780254, 150M cosine | 16,400 / 2,149,580,800 | 3.27632 | 2,535,395 | 79i09kx2 |
| 780284, 300M .002 label | 800 / 104,857,600 | 4.95202 | 1,328,116 | h75v28ld |
| 780285, 300M .0025 label | 200 / 26,214,400 | 8.93603 | 1,331,763 | vp67ndio |

300M Flock URLs:
- https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/h75v28ld
- https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/vp67ndio

All inspected losses/gradients are finite. Three actual 300M group LRs match the
manifest times the exact WSD warmup/plateau factor; .002 is at plateau and .0025
was still in warmup at the recorded step. These asynchronous startup observations
are not final loss comparisons or evidence of long-run stability. No remaining
launch blocker. Next: inspect final dev results after jobs complete.

## Execution authorized and started — 2026-11-15 13:56 (Asia/Taipei)

The user requested execution and explicitly authorized implementing/verifying the
experimental 300M group adapter. This supersedes the configuration-only scope.

- Reused completed 150M power-p1 **778934**, final dev 2.78278, after byte-for-byte
  Python-source and accepted performance-plan comparison. Numerical/data settings
  match; the baseline actual-result metadata confirms all accepted fast flags.
- **780253**, power p=1.25, RUNNING on 25a-hgpn027, eight GPUs / 14 h.
  Flock: https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/qh7g8hkd
- **780254**, cosine, RUNNING on 25a-hgpn051, eight GPUs / 14 h.
  Flock: https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/79i09kx2
- 150M snapshot: `/work/asciibase64/upwash/snapshots/decay150m-20261115T055129Z`.
  Submission record: `/work/asciibase64/upwash/decay150m-pipeline.json`.
- **780263**, full-300M acceptance, RUNNING on 25a-hgpn044, eight GPUs / 2 h.
  Snapshot: `/work/asciibase64/upwash/snapshots/group300m-20261115T055526Z`.
  Record: `/work/asciibase64/upwash/group300m-pipeline.json`.
  Results: `/work/asciibase64/upwash/validation/group300m-780263`.
  Both 60B jobs await successful GPU acceptance. Both retain power p=1.

### Adapter and verification

`experiments/sweep300m/train_grouped.py` consumes the complete manifest, assigns
each trainable parameter exactly once, deduplicates tied weights and constructs
two Muon groups plus AdamW before WSD. LR/WD go directly to Torch without another
width division. It reuses the accepted fast runner in its own process; `src/`
and existing runner code are unchanged. Flock/result metadata records group
values, exhaustive assignment, manifest SHA256 and adapter SHA256.

CPU oracle: installed Torch **2.15.0+cu132** Muon/AdamW with independently assigned
roles, actual forward/backward/updates, both LR rows and a 12-step warmup/plateau/
decay schedule. Tests check independent initial state/inputs, exact group WD,
deduplication, missing/overlapping assignment rejection and parameter agreement
(rtol 1e-6 / atol 1e-7).

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/integration/test_grouped_optimizer.py --tb=short --show-capture=no`
  -> intended Red, **3 failed in 21.42s**: missing adapter module; both real Torch
  reference runs completed before adapter import.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/integration/test_grouped_optimizer.py tests/integration/test_speed_objective.py tests/unit/components/optimizers/test_muon_adamw.py tests/unit/components/schedulers/test_wsd.py --tb=short --show-capture=no`
  -> **19 passed in 13.88s** after implementation.
- Existing runner/config/WSD prelaunch regression: **29 passed in 14.92s**.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh --tb=short --show-capture=no`
  -> **390 passed in 20.47s**.
- Shell syntax checks and `git diff --check` -> exit 0.

GPU gates use full 298,273,792-parameter models, context 4096 and real data.
Native HF objective + independent Torch groups run first, followed by optimized
three-update BF16 comparison for each LR. Existing 150M tolerances are retained:
relative loss < .001, aggregate/tied gradient < .03, update < .10. Each row then
runs 400 actual-Trainer eight-rank updates, four full-dev evaluations, group LRs
ending at zero, and HF export/reload (< .03). `check_acceptance.py` requires
finite metrics and exact step/token accounting. GPU acceptance and long-run
convergence are not yet claimed. No dependencies/test configuration changed;
the unrelated Mirage job remains untouched.

Next: collect GPU acceptance, submit both 60B jobs and verify online startup.

### Acceptance checker correction — 2026-11-15

Both full-model numerical gates pass: LR .002 relative loss/gradient/update
errors .00003366 / .005617 / .013181; LR .0025 .00002440 / .005623 / .013196.
First 400-update integration completed with 52,428,800 tokens, four dev events,
final dev 5.730574, 19.382 GiB peak allocation and export relative error .002932.
The second integration remains in progress at this record.

Inspection exposed a defect in the new acceptance checker: it expected
`train/learning_rate/muon_main`, whereas the established Trainer contract uses
plain `train/learning_rate` for group 0. No training/optimizer change is needed.
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/unit/test_grouped_acceptance.py --tb=short --show-capture=no`
-> **1 failed in .05s**, KeyError for that nonexistent metric. Correcting only the
checker to use the existing key, then running that file plus
`tests/unit/core/test_trainer_clipping_and_lr.py` -> **7 passed in 10.90s**.
The original GPU snapshot is retained; the launch snapshot will differ only in
the corrected checker. Any original batch failure at this final parser is a
verification-helper defect, not a 300M numerical/training failure. All GPU output
gates must still pass through the corrected checker before formal submission.

## Current result — 2026-11-15 (Asia/Taipei)

The user corrected the experiment split: **three 150M decay shapes plus two
300M scaled LR candidates**, replacing the previous six-run 300M cross product.

- [150M / 30B](../experiments/decay150m/README.md): three ordinary RunConfigs
  copied from the completed LR .002 / Adam multiplier 1.6 baseline. Compare
  power p=1, power p=1.25 and cosine; preserve all other numerical/data settings.
  Only shape/power, logging identity and output directory differ from baseline.
- [300M / 60B](../experiments/sweep300m/README.md): retain the two power-p1
  manifests, with 150M-scale LR labels .002 and .0025. Existing correct scaling
  is unchanged: main Muon API LR .0017320508075688772 / .0021650635094610966;
  down_proj .0013685339714124463 / .0017106674642655578; Adam .0032 / .004.
  Remove the four 300M cosine/p=1.25 variants. Shape is fixed at baseline p=1;
  if 150M later selects another shape, apply it identically to both 300M rows.

The 300M files still require per-group LR/WD support and are proposal manifests,
not directly executable RunConfigs. Keep the prior tied-weight/norm adaptation,
independent WD policy and fixed warmup-token convention. No production code,
runner, tests, dependencies or test configuration changed. No jobs submitted.
The original 150M baseline file is unchanged. No new training result is claimed.

### Current verification

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/unit/components/optimizers/test_muon_adamw.py tests/unit/test_residual_scaling.py tests/unit/test_model_config.py tests/unit/test_run_config.py tests/unit/components/schedulers/test_wsd.py --tb=short --show-capture=no`
-> exit 0, **79 passed in 21.41s**. Existing implementation regression only;
no new API or optimizer-group execution coverage is claimed.

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python /tmp/upwash_validate_five_configs.py`
-> exit 0, **all five configuration checks passed** with Torch 2.15.0+cu132.
This temporary read-only validation loads all three direct 150M configs and
both inner 300M RunConfigs, compares 150M settings against the baseline, builds
actual models on the meta device, verifies 149,825,280 / 298,273,792 parameters,
exhaustive non-overlapping group assignment for every parameter, installed
Torch `_adjust_lr` geometry for every matrix role, LR-coupled independent WD,
unique run identities, and token/schedule arithmetic. It performs no training
and is a session diagnostic, not a new permanent repository test.

Full five-run budget: **209,999,495,168 tokens**. Reusing the already completed
150M power-p1 control after matching execution settings leaves four new runs /
**179,999,604,736 tokens**. A configuration count is not a submission count.
Final LR-isolation diagnostic initially asserted scalar scaffold WD equality
and failed: the .0025 manifest intentionally uses .032 instead of .04, keeping
LR-times-WD fixed. Correcting that diagnostic to check the independent-decay
invariant passed; the remaining inner config fields agree after excluding LR,
WD representation and run identity. Authoritative per-group WD likewise varies
inversely with LR. No manifest or production fix was necessary.
`git diff --check` -> exit 0.

### Next action and limits

Configuration delivery is complete. Review the three 150M shapes and the two
300M LR rows. Submission remains outside this task. Before any later 300M
execution, implement and verify the required group support and complete 300M
GPU acceptance; 150M acceptance does not cover that scale. Shape selection is
pending actual 150M results. The earlier proposal and audit below are historical
and do not describe the active file count.

## Historical six-manifest proposal — superseded

User requested transfer of 150M-scale LR [.002, .0025] to 300M / 60B with
power decay p=1 / 1.25 and cosine. After reviewing the optimizer mismatch,
the user explicitly selected **configuration delivery only; no submission**.

Delivered [six manifests and detailed derivation](../experiments/sweep300m/README.md).
They preserve the existing Muon match_rms_adamw definition while applying the
paper's per-matrix geometric scaling. Main Muon LR scales by sqrt(3/4),
down_proj additionally by 64/81. Tied embedding/norm Adam LR remains .0032/.004
under the existing explicit output multiplier; this is an adaptation to the
project architecture. Fix independent WD across LR candidates, anchored to the
completed .002 baseline and scaled by 3/4. Warmup tokens remain fixed; 457,763
updates consume 59,999,911,936 tokens per run, just below nominal 60B.

These are **proposals, not directly executable RunConfig files**. Current
OptimizerConfig/factory cannot represent the necessary per-group LR/WD.
No production API, optimizer, test configuration, dependency, or launcher changed.
No 300M training or submission occurred. The schema is not an agreed API.

## Verification

Existing regression command:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/unit/components/optimizers/test_muon_adamw.py tests/unit/test_residual_scaling.py tests/unit/test_model_config.py tests/unit/test_run_config.py tests/unit/components/schedulers/test_wsd.py --tb=short --show-capture=no`
-> **79 passed in 23.94s**. These tests exercise existing implementation only.
Separate meta-model/read-only validation checks all six inner RunConfigs,
298,273,792 actual parameters, exhaustive non-overlapping group assignment,
Torch's actual LR normalization geometry, plateau independent WD, token budget,
and warmup/decay step counts. No new interface tests were written.

Baseline recheck: `sacct -j 778934 --format=JobID,State,Elapsed,ExitCode -n`
shows COMPLETED / 03:23:04 / 0:0. Its result.json records 228,881 updates,
29,999,890,432 tokens, 12,132.3046 training/eval seconds, and HF export relative
error .00569855; its Slurm log summarizes eval/loss 2.78278. This confirms job
completion, not stability of the new scale or all intermediate baseline metrics.

## Next action and limits

Review these configurations. At the user's request, stop before implementation
or submission. If execution is later requested, agree on an experimental group
adapter and validate it with actual 300M GPU numerical/memory/scheduler/export
checks before full runs. The 150M speed acceptance does not cover 300M.
The 1/width WD heuristic is extrapolated to roughly 200 TPP; optimality and
full CompleteP equivalence (including epsilon) are not established.

## Formula audit — 2026-11-15 (Asia/Taipei)

Rechecked all six actual JSON manifests against the requested relative LR rule.
Using the existing README definition S(A,B)=sqrt(A/B), the rule holds for
Muon's effective update coefficient eta_effective = eta_API * a(A,B), where
installed Torch 2.15.0+cu132 uses a(A,B)=0.2*sqrt(max(A,B)). It does not hold
for the raw API LR alone with that same definition of S.

Read-only Python calculation loaded all six manifests and the 150M baseline,
then evaluated installed Torch's `_adjust_lr` for all seven matrix roles.
Every comparison passed at relative tolerance 1e-12. Effective coefficient
ratios are 1 for q/k/v/o, 9/8 for gate/up, and 8/9 for down; raw API LR ratios
are sqrt(3/4) for the main group and 0.6842669857062231 for down.
The .002 and .0025 labels refer to separate 150M-scale candidate LRs; the
.0025 target is transferred from a hypothetical .0025 baseline, not the
completed .002 baseline.

Adam API LRs remain unchanged relative to each candidate's base LR * 1.6.
The output multiplier ratio is 3/4; tied embedding/norm transfer remains the
previously documented adaptation, not verified universal formula compliance.
The factory still has one Muon group and does not consume the manifests'
required_optimizer_groups. This audit validates proposal arithmetic only.

Existing regression command:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/unit/components/optimizers/test_muon_adamw.py tests/unit/test_run_config.py --tb=short --show-capture=no`
-> **8 passed in 17.94s** (exit 0). No new tests, production changes, config
changes, or GPU runs. Next: review the effective/API LR distinction and the
Adam adaptation before any separately requested implementation.
