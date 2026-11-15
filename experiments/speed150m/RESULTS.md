# Measured 150M speed results

Fixed model: 149,825,280 parameters, seed 2, BF16, FA3, context 4096, global batch 32. All GPU work used Slurm. Versions: Torch 2.15.0, Transformers 5.22.2, Triton 3.9.0, FA3 3.0.1. Liger 0.8.5 was isolated with `--no-deps`; no project dependency upgrade.

## Selected recipe

Use `recommended.json` with `train.sbatch`: eight GPUs, native model+loss compiled in default mode, fused AdamW, pinned prefetch/H2D overlap, and optimizer compilation during the float-LR plateau only. Repeated steady-state median **2,561,742 tokens/s**, versus matched warm baseline **1,973,828** (**29.8% faster**).

Max-autotune peaks at **2,601,377 tokens/s** (**31.8% faster than baseline**), but initial compile+warmup is 305.0 s versus 27.2 s. The additional compile cost breaks even at approximately **46.7B tokens**, using these measured steady-state rates alone. This is an estimate, excludes evaluation compilation, and does not establish completed long-run speed or convergence. Default is the practical recommendation for the current 30B budget.

## Eight-GPU ablations (job 778734, node 25a-hgpn061)

First-pass baseline was much slower and variable; it is retained as raw data but excluded from speedup calculations. Each reported case uses 40 warmup updates and three synchronized 100-update windows, taking maximum elapsed time across ranks. Data iteration, H2D, forward/loss, backward, clipping and optimizer are included. Compilation, evaluation, logging and export are excluded from steady-state rates. Native combined was repeated at the end.

| Variant | Median tok/s | Relative to warm baseline | Allocated / reserved GiB per GPU |
|---|---:|---:|---:|
| baseline-repeat | 1,973,828 | 1.000x | 25.05 / 36.00 |
| fused-adam | 2,011,315 | 1.019x | 25.05 / 36.00 |
| prefetch | 2,045,180 | 1.036x | 25.05 / 36.00 |
| compiled-loss | 2,361,376 | 1.196x | 13.26 / 16.48 |
| native-combined | 2,560,067 | 1.297x | 13.26 / 16.43 |
| native-repeat | 2,563,417 | 1.299x | 13.26 / 16.43 |
| native-max-combined | 2,601,377 | 1.318x | 2.78 / 16.89 |
| liger-fallback | 1,161,771 | 0.589x | 6.17 / 6.44 |
| liger-combined | 1,210,733 | 0.613x | 6.17 / 6.80 |

CUDA-graph pool storage makes allocated memory alone misleading for max-autotune; compare reserved memory as well. The Liger eager-loss boundary works numerically but is slower here. Fully compiling Liger failed with `aten.addmm.dtype / tuned_addmm()` lowering error; it has no valid throughput measurement. No failure was hidden by changing dependencies or relaxing numerical tolerances.

## GPU-count comparison

All rows retain global batch 32 and the same sample grouping. The first three rows use the same default-mode objective+fused-AdamW+prefetch configuration (without optimizer compilation). The final row is the selected eight-GPU recipe including optimizer compilation, explicitly a different configuration.

| GPUs | Local batch | Median tok/s | GPU-hours / billion tokens |
|---:|---:|---:|---:|
| 1 | 32 | 405,271 | 0.685 |
| 2 | 16 | 757,843 | 0.733 |
| 4 | 8 | 1,404,472 | 0.791 |
| 8 (selected recipe) | 4 | 2,561,742 | 0.867 |

Fewer GPUs improve GPU-hour efficiency but take longer in wall time. Four-GPU controls and all raw timing windows are retained under `results/`.

## Full-model and actual-Trainer acceptance

Full-model numerical gates compare a real unfused FA3 reference first, with independent states, real data, full 150M architecture and context 4096. Predeclared limits: relative loss .001, gradient L2 .03, tied-embedding gradient L2 .03, three-update delta L2 .10. The update bound follows the existing BF16 Muon validation; actual errors are reported below.

| Mode / job | Relative loss | Gradient L2 | Update L2 | Actual Trainer status |
|---|---:|---:|---:|---|
| default / 778782 | 0.000034 | 0.005063 | 0.012818 | Passed |
| max-autotune / 778783 | 0.000037 | 0.005093 | 0.012912 | Passed |

Integration uses 400 updates / 52,428,800 tokens, a validation-only WSD warmup .05 / decay .2, four full-dev evaluations, exact final token count, final LR zero, and HF export/reload checks. Both variants have the train/dev file pages prewarmed. The following times include training, compile and evaluation, but exclude model/data construction, export, and the separate numerical gate.

| Mode / job | Baseline seconds | Optimized seconds | Final dev baseline / optimized | Export relative error |
|---|---:|---:|---:|---:|
| default / 778782 | 82.062 | 78.651 | 4.994831 / 4.996673 | 0.003031 |
| max-autotune / 778783 | 82.444 | 543.399 | 4.993171 / 4.995124 | 0.002966 |

Short-run numerical agreement is not proof of 30B stability. Task 028's sustained loss degradation remains unresolved. No replacement 30B training was launched. Existing Mirage work was left untouched.

## Reproduction and evidence

- CPU full suite: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ./test.sh --tb=short --show-capture=no` -> **387 passed in 38.76s** on the final code; the two targeted integration tests also passed after adding explicit graph iteration boundaries.
- Timing scripts: `bench.sbatch`, `refine.sbatch`, `liger.sbatch`, `eight.sbatch`. Acceptance: `accept.sbatch`. All run only under Slurm.
- Raw JSON and exact timing-script snapshots are in `results/<job>/`. Large profiler traces, logs and exported models remain in `/work/asciibase64/upwash/speed150m/<job>/`.
- Exact accepted Python sources are archived under `results/validated-source/`. A subsequent metadata-only correction makes config logging reflect the selected compile mode. Raw max-autotune acceptance metadata retains the original config's `compile_mode=default`; its `speed_optimizations.mode` correctly records the mode actually used. Computation is unchanged.
- Dev QOS permits only eight GPUs/account and two running jobs/user. Eight-GPU jobs were moved to `8gpus` (64 GPUs/account, eight jobs/user); job 778428 was cancelled before start and replaced by 778734.
- A tensor-LR compilation wrapper failed the strict CPU regression and was replaced with plateau-only compilation. The original assertions were retained.
- Recipe applies only to the measured 150M / BF16 / context-4096 / global-batch-32 setup. Other model sizes, batches or hardware require new measurement.
