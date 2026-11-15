# 300M / 60B scaled learning-rate comparison

Status (2026-11-15 14:05, Asia/Taipei): the user now authorized execution and an
isolated experimental adapter. `train_grouped.py` consumes the full manifest;
`src/` and ordinary RunConfig loading are unchanged. Full CPU suite: 390 passed.
Both full-size numerical / eight-GPU WSD/dev/export checks passed. The original
acceptance job 780263 failed only in its final metric-name checker; the corrected
checker passes the retained outputs. Formal 60B jobs **780284** (.002 label)
and **780285** (.0025 label) are RUNNING, eight GPUs each, fixed power p=1.
Online startup is verified for both: finite actual training metrics, correct
scheduled main/down/Adam LRs, and successful Flock uploads (`h75v28ld` / `vp67ndio`).
This supersedes the earlier configuration-only restriction. Detailed evidence
and current job records are in [task 030](../../roadmap/030-300m-sweep-config.md).

## Files and execution boundary

The two JSON files in this directory are **proposal manifests, not executable
RunConfig files**. Use `train_grouped.py`, which consumes the complete manifest.
Each contains a complete `run_config`, the required Torch
optimizer groups, and token accounting. The current loader rejects the outer
manifest. Do not extract `run_config` and launch it alone: the current factory
would ignore the required group distinctions and produce a different experiment.
The group schema is a proposed description, not an agreed production API.

`required_optimizer_groups` is authoritative: its LR and weight_decay values
must replace the factory defaults before scheduler construction. Pass its weight
decays directly to Torch, with no further division by width. Trainable parameters
must be assigned exactly once; tied lm_head/embedding weights are one parameter.
All groups use the same WSD schedule factor. The authorized adapter has CPU
coverage; `accept.sbatch` checks actual 300M numerics, memory, WSD and export
before `train.sbatch` allows formal execution. Online startup is checked for the
formal jobs. The 150M speed recipe is a candidate only;
its acceptance does not establish 300M acceptance.

## Execution

Submit from a frozen source snapshot with `.venv` and credential links to the
existing environment. `sbatch experiments/sweep300m/accept.sbatch` runs both
LR rows through independent native-Torch numerical gates and 400 real Trainer
updates with full-dev evaluation and HF export. Acceptance is offline and does
not consume a formal 60B run.

After acceptance passes, set `MANIFEST` to one complete proposal JSON and
`ACCEPT_RESULTS` to its validated result directory, then submit
`sbatch experiments/sweep300m/train.sbatch` for each LR. The launcher rechecks
acceptance and logs online. Both rows use eight GPUs, batch 32 and power p=1.
Runtime records and exact source hashes live under `/work/asciibase64/upwash/`.
Final HF exports remain the only checkpoints; intermediate resume is unsupported.

## Configuration files and experiment split

The active plan has **five configurations**, with no LR-by-shape cross product:

| Comparison | Model / budget | LR candidates | Decay |
| --- | --- | --- | --- |
| [Decay shape](../decay150m/README.md) | 150M / 30B | Fixed baseline .002 | power p=1, power p=1.25, cosine |
| Scaled LR (this directory) | 300M / 60B | 150M-scale labels .002 and .0025, converted below | Fixed power p=1 |

- [300m-60b-base-lr0.002-power-p1](300m-60b-base-lr0.002-power-p1.json)
- [300m-60b-base-lr0.0025-power-p1](300m-60b-base-lr0.0025-power-p1.json)

The four former 300M cosine/p=1.25 proposals have been removed. The 300M pair
currently retains the baseline's power p=1 schedule, so only LR varies. If the
150M comparison later selects a different shape, apply that same shape to both
300M candidates before execution; this is not an additional shape sweep.
No shape winner or new training outcome is claimed.

## Reference and baseline

Paper: [Qiu et al., arXiv:2512.05620v2](https://arxiv.org/pdf/2512.05620v2),
Table 1, Sections 3.1–3.4 and H.3. Apply relative per-matrix update scaling,
not a rule based on total parameter count. This is an adaptation to Upwash's
existing parameterization, not a reproduction of the paper's full recipe.

Baseline: `configs/base150m/base150m-30b-seed2-lr0.002-am1.6.json`.
Job 778934 is COMPLETED (exit 0:0); its result records 228,881 updates and
29,999,890,432 tokens, and its Slurm log summarizes final dev loss 2.78278.
This does not establish stability of the proposed 300M runs or LR=0.0025.

Inspected implementation: Torch 2.15.0 Muon, `match_rms_adamw`, with raw API LR
eta and update multiplier `a(A,B) = 0.2 * sqrt(max(A,B))` for W[A,B]. Its
weight shrinkage is `1 - eta * weight_decay`, using **raw**, not adjusted LR.
Upwash currently passes `OptimizerConfig.weight_decay / d_model` to both
Muon and AdamW. Thus the existing 0.04 is not independent weight decay.

## Architecture and initialization

| Setting | 150M baseline | 300M target |
| --- | ---: | ---: |
| Exact parameters | 149,825,280 | 298,273,792 |
| Layers | 6 | 9 |
| d_model | 768 | 1024 |
| d_ff | 3072 | 5184 |
| Query / KV heads | 12 / 3 | 16 / 4 |
| Head dimension | 64 | 64 |
| Residual branch multiplier | 1/6 | 1/9 |
| Vocabulary / context | 128256 / 4096 | unchanged |
| Tied embedding std | 0.1 | unchanged |
| output_std_coeff | 1.0 | unchanged |
| Explicit logit multiplier | 1/(768*0.1) | 1/(1024*0.1) |

Keep `residual_scale_by_depth=true`. Existing initialization automatically uses
`in_features**-0.5`: no global initialization override is necessary. In particular,
down_proj initialization scales by sqrt(3072/5184), rather than sqrt(768/1024).
The effective tied readout std remains 1/d_model through the output hook.

## Learning-rate conversion

For each matching matrix role, preserve the base effective update while applying
paper Muon scaling S(A,B)=sqrt(A/B):

    eta_target / eta_base = (S_target / S_base) * (a_base / a_target)

This retains Torch's existing normalization and finite Newton–Schulz update,
while correcting the relative geometric scaling. It does not turn approximate
Newton–Schulz into an exact spectral normalization algorithm.

- q/k/v/o, gate/up: raw LR multiplier sqrt(768/1024) = 0.8660254037844386.
- down_proj: raw LR multiplier 0.6842669857062231. Relative to the main target
  group this is 64/81, because d_ff/d_model changes from 4 to 5.0625.
- Keep tied embedding and RMSNorm Adam API LR at base_lr * 1.6. The embedding
  input vocabulary does not scale. The explicit output multiplier already
  scales the effective readout LR by 768/1024; applying that factor again to the
  tied weight's API LR would double-count the readout transfer. Constant norm
  LR is the chosen extension for this existing architecture. These shared tied
  weights and trainable norms differ from the paper's main experimental setup;
  their transfer remains an empirical hypothesis.
- Consequently the run_config Adam multiplier is 1.6/sqrt(768/1024) =
  1.8475208614068026, relative to its reduced main Muon LR.
- Keep Muon momentum .9, Adam betas (.9,.99), existing optimizer epsilon defaults,
  and clipping 1.0. Epsilon effects are not part of the geometric conversion;
  this is not a claim of exact all-hyperparameter CompleteP equivalence.

| 150M-scale LR label | Main Muon API LR | down_proj API LR | Adam API LR |
| --- | ---: | ---: | ---: |
| .002 | .0017320508075688772 | .0013685339714124463 | .0032 |
| .0025 | .0021650635094610966 | .0017106674642655578 | .0040 |

Both rows use power p=1: **two 300M runs total**. The three decay shapes are
compared only at 150M. In this scheduler power means `(1-progress)**p`; cosine
means `0.5*(1+cos(pi*progress))`, ending at zero.

## Weight decay: fix independent decay across LR candidates

Anchor to the completed .002 baseline, then scale independent decay by
768/1024. Keep these plateau shrinkage coefficients fixed across both LR
candidates, so LR is the only optimizer quantity swept:

    muon_lambda = (.002 * .04 / 768) * (768/1024) = 7.8125e-8
    adam_lambda = (.0032 * .04 / 768) * (768/1024) = 1.25e-7
    torch_weight_decay_group = lambda_group / plateau_lr_group

| 150M-scale LR label | Main Muon Torch WD | down_proj Torch WD | Adam Torch WD |
| --- | ---: | ---: | ---: |
| .002 | .000045105489780439515 | .00005708663550336876 | .0000390625 |
| .0025 | .00003608439182435161 | .00004566930840269501 | .00003125 |

During warmup/decay, lambda(t)=schedule_factor(t)*lambda_plateau. This preserves
the baseline convention of scheduling decay with LR. It is not constant decay
at every training step. Different decay shapes intentionally change integrated
decay. The manifest's scalar optimizer.weight_decay is merely a compatibility
scaffold; the per-group values above are required for correct execution.

The 1/width independent-WD rule is an empirical extrapolation here: this project
trains about 200 tokens/parameter, well beyond the paper's main 20-TPP scaling
setup. These numbers are controlled sweep candidates, not proven optimum WD.

## Shared runtime and schedule

- Same train/dev dataset paths, seed 2, data shuffle seed 0, tokenizer and revision
  as baseline. BF16, FA3, candidate default-mode compiled objective, fused AdamW,
  prefetch and plateau-only optimizer compilation.
- Eight GPUs, local batch 4, global batch 32, 4096 tokens/sequence:
  131,072 tokens/update; no gradient accumulation.
- Requested 60,000,000,000 tokens; round down to full global updates as baseline:
  **59,999,911,936 tokens / 457,763 updates**, 88,064 tokens below 60B.
- `warmup_ratio=.0005`: 228 updates, matching the baseline's 228 warmup updates
  (29,884,416 tokens) rather than doubling warmup tokens with the training budget.
- `decay_ratio=.1`: 45,776 decay updates; plateau ends at update 411,987.
- Log every 100 updates; evaluate full dev every 22,888 updates (about 3B tokens),
  plus final evaluation. Eval local batch 8.
- The two 300M candidates total **119,999,823,872 training tokens**. The three
  150M configurations total **89,999,671,296**, for **209,999,495,168** across
  five full runs. The completed 150M power-p1 baseline may be reused as that
  control after confirming the execution recipe matches; then four new runs
  would require **179,999,604,736** tokens. No 300M throughput
  estimate or queue/runtime guarantee is claimed from the 150M benchmark.
- Existing training saves final HF export only; no intermediate checkpoint/resume
  capability is added by these proposal files.

## Verification

Current five-config validation and regression results are recorded in
[task 030](../../roadmap/030-300m-sweep-config.md). Validation covers direct 150M
RunConfig loading, baseline isolation, inner 300M RunConfigs, exact model counts,
non-overlapping exhaustive optimizer groups, Torch normalization conversion,
independent decay, and token/schedule accounting. It does not establish 300M
GPU acceptance or production support for the required optimizer groups.
