# 150M Sweep Results (overnight driver)

Rules: waves of <=10, next wave after the whole wave ends; best = lowest final eval/loss; stop on edge lr, code error, or after S3.

## Dev exclusion check (2026-11-14 06:2x, agent, driver paused)

Pool manifest reports `dev_set.excluded = 0`, which is plausible only if no
pool file holds dev documents. `cerulean-works/fineweb-edu-dev` `dev-set.json`
and its README say that all 5000 dev documents were sampled from one file,
`data/CC-MAIN-2025-26/000_00000.parquet`. The pool uses 7 files of that dump
(000_00010, 28, 29, 32, 38, 48, 49) but not 000_00000, so 0 is expected. The
`dev-ids.txt` ids equal the dev parquet `id` column and use the training
`id` format (`<urn:uuid:...>`). The driver was stopped before any submission
and restarted after this check.
## Pool verification (2026-11-14 06:18)

- total_tokens 205,087,998,552, docs 203,705,949, shards 326, dev ids excluded 0, seed 0
- problems: none


**Stopped:** s1-lr0.032-mb0.98 (job 773894) failed: `COMPLETED` exit `0:0`.

```
ng model shards: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s]
wandb: updating run metadata
wandb: 
wandb: Run history:
wandb:                 eval/loss ▁▃▅▇█      
wandb:           eval/perplexity ▁▃▅▇█      
wandb:           train/grad_norm ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█               
wandb:       train/learning_rate ███████████████████████████████████▆▂▂▂▁
wandb: train/learning_rate/adamw █████████████████████████████████████▆▄▁
wandb:                train/loss ▁▁▁▁▁▁▁▂▁▂▂▂▂▂▂▂▂▂▃▂█                   
wandb:   train/tokens_per_second ▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁██████████████████
wandb:   train/total_tokens_seen ▁▁▁▁▁▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇█████
wandb: 
wandb: Run summary:
wandb:                 eval/loss nan
wandb:           eval/perplexity nan
wandb:           train/grad_norm nan
wandb:       train/learning_rate 0
wandb: train/learning_rate/adamw 0
wandb:                train/loss nan
wandb:   train/tokens_per_second 1748667.93721
wandb:   train/total_tokens_seen 2999975936
wandb: 
wandb: 🚀 View run s1-lr0.032-mb0.98 at: https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/ty38hwy9
wandb: ⭐️ View project at: https://flock-api.cerulean.works/cerulean-labs/lab6-predictable-scaling
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: /work/u7725908/upwash/wandb/wandb/run-20261114_072407-ty38hwy9/logs
shutdown can only be called once
[07:54:58 +1878s] done: /work/u7725908/upwash/runs/s1-lr0.032-mb0.98

```


## Driver fix (2026-11-14 08:00, agent)

The stop above was a driver bug, not a code error: job 773894 completed with
exit 0 but diverged to NaN, and the summary regex did not match `nan`, so the
run was classified as `code`. NaN runs now count as a valid result with
`eval/loss = inf` and `diverged: true`. The stop flag was cleared and the
driver restarted; it re-classifies 773894 and resumes 773899/773900.

## S1: lr x Muon beta (beta1 0.85, beta2 0.95, multiplier 1.5)

| run | eval/loss | eval PPL | train/loss | tok/s |
|---|---|---|---|---|
| s1-lr0.004-mb0.9 | 2.9614 | 19.33 | 3.1637 | 1,528,348 |
| s1-lr0.004-mb0.95 | 2.9735 | 19.56 | 3.1761 | 1,708,013 |
| s1-lr0.002-mb0.9 | 2.9779 | 19.65 | 3.1844 | 1,547,598 |
| s1-lr0.002-mb0.95 | 2.9798 | 19.68 | 3.1812 | 1,630,545 |
| s1-lr0.002-mb0.98 | 3.0551 | 21.22 | 3.2580 | 1,703,134 |
| s1-lr0.002-mb0.99 | 3.2638 | 26.15 | 3.4642 | 1,647,623 |
| s1-lr0.008-mb0.95 | 4.4433 | 85.06 | 4.5946 | 1,592,751 |
| s1-lr0.016-mb0.9 | 4.4590 | 86.40 | 4.6115 | 1,588,432 |
| s1-lr0.008-mb0.9 | 4.4990 | 89.92 | 4.6693 | 1,696,425 |
| s1-lr0.004-mb0.98 | 4.5969 | 99.18 | 4.7556 | 1,619,477 |
| s1-lr0.004-mb0.99 | 4.6641 | 106.07 | 4.8090 | 1,708,563 |
| s1-lr0.008-mb0.98 | 5.2326 | 187.28 | 5.3450 | 1,520,259 |
| s1-lr0.016-mb0.98 | 5.3292 | 206.27 | 5.4240 | 1,674,859 |
| s1-lr0.008-mb0.99 | 5.4009 | 221.60 | 5.4965 | 1,690,972 |
| s1-lr0.016-mb0.95 | 5.6204 | 276.01 | 5.7093 | 1,623,487 |
| s1-lr0.032-mb0.9 | 6.6065 | 739.90 | 6.6167 | 1,653,992 |
| s1-lr0.032-mb0.95 | 9.0116 | 8197.91 | 8.9243 | 1,630,977 |
| s1-lr0.032-mb0.98 | inf | nan | nan | 1,748,668 |
| s1-lr0.016-mb0.99 | inf | nan | nan | 1,744,418 |
| s1-lr0.032-mb0.99 | inf | nan | nan | 1,746,575 |

S1 best: **s1-lr0.004-mb0.9** (lr 0.004, Muon beta 0.9).

## S2: AdamW beta1 and seed replica

| run | eval/loss | eval PPL | train/loss | tok/s |
|---|---|---|---|---|
| s2-noise-lr0.004-mb0.9-seed1 | 2.9586 | 19.27 | 3.1606 | 1,705,797 |
| s2-lr0.004-mb0.9-b1_0.9 | 2.9592 | 19.28 | 3.1599 | 1,518,161 |
| s1-lr0.004-mb0.9 | 2.9614 | 19.33 | 3.1637 | 1,528,348 |

Seed-to-seed |delta eval/loss| = 0.0028. beta1 chosen: **0.9** (0.9: 2.9592 vs 0.85: 2.9614).

## S3: AdamW beta2 x LR multiplier

| run | eval/loss | eval PPL | train/loss | tok/s |
|---|---|---|---|---|
| s3-lr0.004-mb0.9-b1_0.9-b2_0.99-m1.6 | 2.9356 | 18.83 | 3.1332 | 1,662,310 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.99-m1.5 | 2.9399 | 18.91 | 3.1430 | 1,694,674 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.99-m1.7 | 2.9409 | 18.93 | 3.1442 | 1,632,223 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.98-m1.6 | 2.9425 | 18.96 | 3.1428 | 1,646,589 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.98-m1.7 | 2.9428 | 18.97 | 3.1463 | 1,587,437 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.98-m1.5 | 2.9428 | 18.97 | 3.1402 | 1,448,984 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.95-m1.7 | 2.9570 | 19.24 | 3.1593 | 1,628,154 |
| s2-lr0.004-mb0.9-b1_0.9 | 2.9592 | 19.28 | 3.1599 | 1,518,161 |
| s3-lr0.004-mb0.9-b1_0.9-b2_0.95-m1.6 | 2.9621 | 19.34 | 3.1636 | 1,540,074 |

S3 best: **s3-lr0.004-mb0.9-b1_0.9-b2_0.99-m1.6**. Stopped after S3 as agreed; warmup/decay/shape/weight decay await review.

