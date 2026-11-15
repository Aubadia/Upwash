# 029 — Measured training speed optimization

## Complete — final result, 2026-11-15

All six jobs (778431, 778509, 778701, 778734, 778782, 778783) are COMPLETED.
Measured all five authorized optimization categories and integrated the winning
combination in the opt-in experimental runner. Eight GPUs, global batch 32:
matched warm baseline **1,973,828 tokens/s**; repeated default combination
**2,561,742 (+29.8%)**; max-autotune combination **2,601,377 (+31.8%)**.
Default mode is recommended for the current 30B budget: max-autotune adds about
278 seconds of initial compilation for only about 1.55% more steady throughput.
The estimated break-even is 46.7B tokens, excluding evaluation compilation.
Liger fallback passes numerical checks but is slower, so it is not selected.

Both default and max-autotune passed full-150M/4096-context real-data numerical
gates, followed by 400 actual-Trainer updates (52,428,800 tokens), WSD warmup and
decay to zero, four full-dev evaluations, and HF export/reload checks. Default
training+evaluation including compilation took 78.651 s versus baseline 82.062 s;
max-autotune took 543.399 s versus baseline 82.444 s. These short-run timings are
distinct from steady-state throughput. Final dev losses were 4.994831/4.996673
(baseline/default), and 4.993171/4.995124 (baseline/max-autotune).

Artifacts: [results](../experiments/speed150m/RESULTS.md), raw JSON and validated
source snapshots, both accepted recipe JSON files, and explicit `train.sbatch`.
No formal 30B run was submitted. The unrelated Mirage job was left untouched.
Final queue inspection found no remaining jobs for this user.

Full CPU command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh --tb=short --show-capture=no`.
Final verification: **387 passed in 38.76s**; `git diff --check` also passes.
After GPU acceptance, the runner's config logging was normalized to the actual
selected compile mode; compilation already used that mode, so computation is
unchanged. Exact accepted source is archived in `results/validated-source/`.

Next action: investigate task 028's long-run loss instability. Short numerical
agreement and speed improvements do not establish 30B convergence.

## Historical progress (superseded by the completed result above)

## Default recipe accepted — 2026-11-15 02:07

Job **778782 COMPLETED**. Both actual-Trainer runs completed 400 updates /
52,428,800 tokens, evaluations at 100/200/300/400, final Muon LR 0, finite
logged metrics, and HF serialization/reload checks. Baseline training+evaluation
time (including compilation): **82.062 s**; optimized **78.651 s**. This short
integration improvement is distinct from steady-state throughput. Final dev
losses are **4.994831** / **4.996673**; export relative errors **.002938** /
**.003031**. Full-model numerical gate details follow below.

Saved `experiments/speed150m/recommended.json` and explicit `train.sbatch`
launcher for eight GPUs/global batch 32. It uses native compiled objective,
fused AdamW, prefetch, and plateau-only optimizer compilation; Liger is not
enabled. Default is provisionally preferred for 30B because max-autotune gains
only ~1.6% steady throughput at an extra ~278 s initial compile/warmup cost.
The max-throughput option and final warm baseline repeat are still being
validated/measured. No formal 30B run has been submitted.

## Full-size acceptance in progress

Eight-GPU screen 778734: compiled objective 2,361,376 tokens/s; default
objective + fused AdamW + prefetch + compiled optimizer 2,560,067 tokens/s.
The first baseline is cold and is excluded from speedup claims; the matched
warm repeat is still pending. The Liger fallback works but its four-GPU
combined throughput is only 984,056 tokens/s versus native 1,427,145.

Full-model acceptance jobs **778782** (default, 25a-hgpn075) and **778783**
(max-autotune, 25a-hgpn112) are RUNNING on 8 GPUs each in `8gpus`. Default's
full-150M, real-data, 4096-context numerical gate passed: relative loss
.0000341, gradients .005063, tied embedding gradients .004632, three-update
delta .012818. Actual Trainer/WSD/dev/export acceptance is in progress.
Both integration variants prewarm the same train/dev file pages before timing.

## Scheduling correction — 2026-11-15 01:54

The user asked why 8 GPUs were not used and whether dev would help. Inspection
of actual Slurm QOS settings showed that the restriction is **partition-specific**:
`dev` uses `p_dev` (MaxJobsPU=2, MaxTRESPA=gres/gpu=8), while `8gpus` uses
`p_8gpus` (MaxJobsPU=8, MaxTRESPA=gres/gpu=64). The earlier broad statement about
an account-wide eight-GPU limit was inaccurate. The benchmark already used dev.
Cancelled pending job 778428 and replaced it with focused eight-GPU job **778734**
in partition `8gpus`; it is **RUNNING on 25a-hgpn061**. Mirage 778244 remains
untouched. Four-GPU refinement 778509 completed; fallback/finalists job 778701
is running on 25a-hgpn004. Eight-GPU acceptance will also use `8gpus`.

## Authorized scope — 2026-11-15

The user authorized measuring and combining five optimizations: fused LM head
and cross-entropy, fused AdamW, expanded compilation/modes, prefetch/H2D overlap,
and GPU/local-batch choices. Subsequently approved isolated, version-pinned
Liger without upgrading Torch/Transformers or changing project dependencies.

Implementation and measurement live in `experiments/speed150m/`. Liger 0.8.5
was installed with `uv pip install --python .venv/bin/python --target
/work/asciibase64/upwash/speed150m-deps/liger-0.8.5 --no-deps liger-kernel==0.8.5`.
Reference versions, numerical gates and measurement protocol are documented in
that directory's README. This work does not authorize another 30B calibration.

## Progress

- Inspected the current model, loss, optimizer, data path and actual histories.
  The historical pre-onset rank-zero throughput median is 1,963,587 tokens/s;
  it is not a synchronized benchmark and is not the benchmark's speedup baseline.
- Created an isolated real-data benchmark with synchronized window timing,
  fresh model/optimizer state per candidate, all-rank timing/memory reduction,
  global batch 32, three timing windows, and a real BF16 differential gate.
- Syntax and `git diff --check` pass. CPU baseline:
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh` -> **385 passed in 37.83s**.
- Eight-GPU job 778428 is PENDING (MaxGRESPerAccount). An unrelated two-GPU
  Mirage job remains untouched. Four-GPU screen **778431** is RUNNING on
  25a-hgpn019. Tiny real GPU gates pass: fused AdamW gradient relative L2 0,
  update .00320; fused loss gradient .00658, tied embedding gradient .00686,
  update .00940, max relative loss .0000907. These are short numerical gates,
  not throughput or convergence results. Full-size timing is underway.
- Added opt-in `train_fast.py`, composing the existing Trainer, WSD and HF
  export with benchmark flags. Its actual-Trainer CPU regression, including
  repeated evaluation and tied-weight updates, passes:
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh tests/integration/test_speed_objective.py`
  -> **1 passed in 12.82s**. Full-model GPU acceptance remains pending.
- Refinement job **778464** was cancelled before start and replaced by **778509**
  to include the optimizer-compile combination. It is RUNNING and compares
  4/2/1 GPUs at fixed global batch 32. First screen results suggest compiling
  the complete objective substantially reduces peak allocation (47.77 to
  24.19 GiB at local batch 8). Do not use the first cold-data baseline to claim
  a final speedup: warm baseline repeats and longer measurements are pending.

## First screen and integration checks

Screen 778431 completed. Warm baseline repeat: 1,136,216 tokens/s; first
cold-data baseline: 769,569 (excluded from final speedup claims). Compiled
objective: 1,391,123, fused AdamW: 1,148,457, prefetch: 1,149,196, compiled
optimizer: 1,159,190. Reduce-overhead: 1,089,902; max-autotune: 1,105,888 with
298.56 s initial compile/warmup. All are 4 GPUs, global batch 32. These are
screening measurements, not final integration results.

Compiled Liger and its combined variant failed with Inductor lowering error
`aten.addmm.dtype: tuned_addmm() takes 3 positional arguments but 4 were given`.
No throughput is assigned to them. An explicit eager-Liger-loss boundary keeps
the backbone compiled; fallback job **778603** was cancelled before start and
replaced by **778701** to include max-autotune plus compiled optimizer. It will
compare against native compiled-objective controls on the same node.

The attempted tensor-LR wrapper failed the strict CPU regression (maximum
parameter error about 2.5e-5); float64 LR tensors did not fix it. Assertions were
not relaxed. Replaced it with compilation only at the original float-valued LR
plateau and eager updates during warmup/decay. The same CPU file now gives
**2 passed in 10.61s**. Actual GPU schedule acceptance is still required.

Full CPU regression after that correction:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh --tb=short --show-capture=no`
-> **387 passed in 44.19s**. The full numerical and integration GPU acceptance
has not yet run. Refinement's compiled-objective + fused AdamW + prefetch +
compiled optimizer candidate measures 1,423,789 tokens/s at 4 GPUs, with a
passing small GPU gate (gradient relative .00901, update .01123). This is
provisional until the remaining comparisons and actual-Trainer acceptance.

Refinement: max-autotune with compiled objective reaches 1,436,355 tokens/s,
with 318.85 s initial compile/warmup. It must be compared with the compiled
optimizer combination before selecting a long-run recipe. Default native
objective + fused AdamW + prefetch at 2 GPUs / local batch 16 reaches 757,843
tokens/s and .733 GPU-hours/B tokens (4-GPU counterpart .791). Global batch
remains 32. CUDA-graph allocated-memory statistics exclude some retained pool
storage, so reports include reserved memory as well.

## Next

Collect numerical gates and individual timing results; reject failed candidates.
Measure combinations and GPU/local-batch alternatives, verify the winning
configuration in the actual Trainer with LR scheduling/evaluation/export, and
record measured speedups and remaining stability limitations. No winner yet.
