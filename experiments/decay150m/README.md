# 150M / 30B decay-shape comparison

Status (2026-11-15 13:56, Asia/Taipei): execution authorized. Completed power-p1
control 778934 is verified reusable; power-p1.25 **780253** and cosine **780254**
are RUNNING on eight GPUs each. Final shape results remain pending.
This is the three-shape part of [task 030](../../roadmap/030-300m-sweep-config.md).
The other part is the [two scaled 300M learning rates](../sweep300m/README.md).

## Configurations

These are ordinary, directly loadable RunConfig JSON files copied from
`configs/base150m/base150m-30b-seed2-lr0.002-am1.6.json`. Only decay shape/power,
logging identity and output directory differ. No 300M scaling is applied here.

| Configuration | Decay factor during the final 10% |
| --- | --- |
| [power p=1](150m-30b-lr0.002-power-p1.json) | `(1-progress)` |
| [power p=1.25](150m-30b-lr0.002-power-p1.25.json) | `(1-progress)**1.25` |
| [cosine](150m-30b-lr0.002-cosine.json) | `0.5*(1+cos(pi*progress))` |

All three retain Muon API LR **.002**, Adam multiplier **1.6** (LR **.0032**),
Muon momentum .9, Adam betas (.9,.99), weight_decay .04 (factory divides by
width 768), clip 1, seed 2, data shuffle seed 0, architecture, initialization,
train/dev paths, evaluation frequency and token budget. Different decay shapes
also change integrated LR-coupled weight shrinkage; this matches the existing
optimizer/scheduler convention.

Eight GPUs with local batch 4 and context 4096 give 131,072 tokens/update.
Each config uses **228,881 updates / 29,999,890,432 tokens**, warmup ratio .001
(**228 updates**), decay ratio .1 (**22,888 updates**), with decay starting after
update 205,993. Model parameters: **149,825,280**. The total for three full runs
is **89,999,671,296 tokens**. Distinct logging names/output directories avoid
collisions. The baseline JSON itself is unchanged.

For comparable execution, use the completed baseline's accepted default-mode
objective recipe: FA3, compiled objective, fused AdamW, prefetch and plateau-only
optimizer compilation (`--scope objective --mode default --fused-adam --prefetch
--compile-optimizer`). These flags are execution metadata, not new RunConfig
fields. Merely loading a config does not select that external runner recipe.

Completed job 778934 already supplies the power-p1 baseline result (final dev
loss 2.78278). Source/config/plan comparison now confirms the same execution
recipe; it is reused without a new power-p1 run. The p=1.25 and cosine outcomes
are unknown. The 300M pair currently fixes power p=1; if a different shape wins,
set that shape identically on both candidates before execution.

Current validation evidence is in task 030. Direct config loading does not
constitute GPU execution or submission authorization.
