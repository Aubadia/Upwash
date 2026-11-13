# 026 — Runnable Sweep Entry: CLI, DDP, Accumulation, Export, Data Pool

## Ownership Note — 2026-11-14

Short on time, the user explicitly asked the agent to write the production code
needed to run experiments ("有辦法實驗的部分你先幫我 vibe 一下"). This task is
therefore agent-implemented, on branch `feat/run-entry` from `main` at
`1faf6dc`. Not committed. The user should review it before merging.

## Behavior Added

- **CLI and JSON configs.** `python -m upwash.train --config <json> [--name] [--output-dir]`.
  `RunConfig.to_dict()` and `RunConfig.from_dict()` round-trip through
  `load_run_config(path)`. Unknown keys raise `TypeError`. Examples:
  `configs/smoke-150m.json` and `configs/sweep-150m-base.json`.
- **DDP.** Under `torchrun`, `WORLD_SIZE > 1` initializes NCCL (CUDA) or gloo
  (CPU) and wraps the model with `forward_sync_buffers=False`. `RankShard`
  assigns rank r the r-th block of each global batch and rejects datasets that
  don't fill whole global batches. Only rank 0 logs and exports. Every rank
  evaluates the full dev set, with no collectives.
- **Gradient accumulation.** `TrainingConfig.micro_batch_size` splits each
  per-rank batch. Micro-losses are weighted by sample share, and DDP `no_sync`
  is used for all micro-batches except the last.
- **Clipping.** `TrainingConfig.max_grad_norm` clips the gradients, and
  `train/grad_norm` stays the pre-clip value.
- **Per-group LR.** With more than one param group,
  `train/learning_rate/group_{i}` is logged as well. `train/learning_rate`
  stays group 0, which is Muon.
- **Global token metrics.** `Trainer(world_size=W)` multiplies counted tokens
  by W. This is exact only for unpadded, equal-shaped batches, which
  `RankShard` guarantees.
- **wandb config.** It now holds the full `RunConfig.to_dict()` plus `derived`:
  world size, global batch size and tokens, total steps, planned train tokens,
  parameter count, initialization description, and git commit. This replaces
  the earlier `config={}` assertion in `tests/unit/test_train.py`, because the
  lab requires these fields.
- **`use_cache=False`** in `config_adaptor`.
- **Export.** `export_hf_model(model, config, max_position_embeddings=None)`
  follows the design in task 025: a fresh HF Llama loads the weights, and α is
  folded into `model.norm.weight`. `save_model` writes the model and, when
  configured, the tokenizer. The tokenizer is loaded before training, so a
  missing offline cache fails early. `export_max_position_embeddings=8192`
  changes no logits, because default RoPE ignores it. This still awaits the
  TA's answer on Flock's 8192 requirement.
- **Sample-level shuffle.** `DataConfig.shuffle_seed` applies a seeded
  `Permute` over the whole pool before `Truncate`. Shorter runs therefore read
  a prefix of the same order.
- **Data pool.** `scripts/prepare_pretrain.py` reads HF
  `HuggingFaceFW/fineweb-edu@87f0914…` files in a seeded order, mixing the 110
  dumps. It runs spawn workers, writes `[BOS, text, EOS]` uint32 shards,
  excludes dev ids, deletes raw parquet, stops at `--target-tokens`, and
  writes `manifest.json`. Reruns reuse finished shards.
- **Slurm.** `scripts/train.sbatch`: `torchrun` on 8 GPUs, lab5/midterm node
  exclusions, `NCCL_SOCKET_IFNAME=lo`, `HF_HUB_OFFLINE=1`, and Flock wandb.

## Verification — 2026-11-14 (login node, CPU)

- Full suite: `386 passed in 87.20s`.
- New tests:
  - `tests/unit/test_run_config.py`, `tests/unit/core/test_permute_and_rank_shard.py`,
    and `tests/unit/core/test_trainer_accumulation.py`: accumulated updates
    equal full-batch updates for micro sizes 1, 2, 3, 4, 6.
  - `tests/unit/test_export.py`: reloaded logits match the training logits.
  - `tests/unit/test_prepare_pretrain.py`.
  - `tests/integration/test_ddp_training.py`.
- DDP test: 2 gloo ranks (batch 2, micro 1) through `train()` against one
  process (batch 4). The ranks are bit-identical. AdamW params agree to about
  1e-6 and Muon matrices to about 1e-4, because `torch.optim.Muon`
  orthogonalizes in bf16 (`torch/optim/_muon.py:56`). The tolerance is
  `atol=5e-4`, with a ≥10× training-change guard.
- CPU `torchrun --nproc_per_node 2` end-to-end on a tiny 128k-vocab Llama: it
  reads the probe bucket, runs dev evaluation, exports, and the export reloads
  with `from_pretrained`. `max_position_embeddings` is 8192 and BOS is 128000.
- The tokenizer is cached for offline jobs in
  `/work/u7725908/.cache/huggingface`, and `HF_HUB_OFFLINE=1` loading works.
  `scripts/prepare_dev.py` was rerun to `/work/u7725908/fineweb-edu-dev-tokens/`,
  byte-identical to the earlier output.

- Real-data pool run (login node, 1 spawn worker, through `prepare()`): ran on
  the smallest HF file, `data/CC-MAIN-2024-51/000_00049.parquet` (575 MB).
  Output: 190,532,927 tokens and 194,748 docs in 573 s, about 0.33M tok/s per
  worker including the download. Checks: 194,748 BOS and 194,748 EOS; the
  shard starts with BOS, ends with EOS, and has max id 128001. 0 dev ids
  excluded in this file. The manifest records source/tokenizer/dev revisions
  and the seed.
- Estimate: 205B tokens ≈ 170 worker-hours, about 1.8 h on 96 workers if HF
  download bandwidth (about 90 MB/s aggregate) holds. That bandwidth is
  unverified.

## GPU Smoke — Job 773184 (2026-11-14)

The code was uncommitted, so the logged commit `1faf6dc` is not the code that
ran.

- Command: `CONFIG=configs/smoke-150m.json sbatch -J smoke scripts/train.sbatch`.
  It ran on `dev`, node `25a-hgpn078`, 8×H200.
- Result: `COMPLETED`, exit 0, 69 s wall time for the whole job.
- Training: 150M, 60 updates, global batch 128×4096 = 524,288 tokens,
  `micro_batch_size=8`, bf16, `max_grad_norm=1.0`. Data was one bucket of the
  other pair's shuffled set. No OOM.
- Flock online logging from the compute node works (run `6wa64jkg`). Metrics
  include `train/learning_rate/group_0/1`, and `eval/loss` was logged twice.
- Summary values: `train/total_tokens_seen` 31,457,280 (exact);
  `train/tokens_per_second` 2.02M (about 253k per GPU); final `train/loss`
  9.677; `eval/loss` 9.664 (PPL 15,738); final `train/grad_norm` 0.063.
- Export: `/work/u7725908/upwash/runs/smoke-150m-773184/` contains
  `model.safetensors` (599 MB) and the tokenizer.
- Not checked: the initial loss against ln V ≈ 11.76 (the Flock server rejects
  `wandb.Api` queries: "operation None is not supported"), and peak memory,
  which is not logged.
- Projection: 150M for 30B tokens at 2.02M tok/s ≈ 4.1 h ≈ 33 H200-hours,
  within the 40 h grant. The 3B-token sweep base ≈ 25 min plus evaluation.

## Logged Fields Trimmed — 2026-11-14

The user pointed to the lab's logging section: send what it requires, not
runtime plumbing. Smoke run `6wa64jkg` had sent 9 metrics, including a
duplicate `train/learning_rate/group_0`, and 59 config leaves (the whole
`RunConfig`). Changes:

- Metrics: `train/learning_rate` is group 0 (Muon). Each further group adds
  `train/learning_rate/<group name>`. `build_muon_adamw` names its groups
  `muon` and `adamw`, so runs log `train/learning_rate/adamw`. Unnamed groups
  fall back to their index. The run total is 7 required metrics plus 1.
- Config, 40 leaves: `model` (plus `num_parameters`), `initialization`
  (embedding std, hidden std rule, numeric α), `optimizer` (plus
  `max_grad_norm`, `precision`), `batch`, `schedule`, `duration`
  (`total_steps`, `train_tokens` = the budget, `seq_len`, `eval_steps`),
  `data` (train/eval roots, shuffle seed), `seed`, `git_commit`. Dropped:
  device, eval batch size, output/tokenizer/export settings, logging names.
  Fixed: `train_tokens` was steps × global batch, which overcounts a partial
  final batch.
- Smoke runs: use `WANDB_MODE=offline` (documented in `scripts/train.sbatch`).
  Run `6wa64jkg` remains in the shared project. The Flock server rejects
  `wandb.Api`, so it must be deleted in the web UI.
- Full suite: `387 passed in 93.24s`.

## Gradient Accumulation Removed — 2026-11-14

To keep the agent-written share small, the user decided against
accumulation for now: "minibatch 先不用".

- Removed: `TrainingConfig.micro_batch_size` and its validation, the
  `Trainer` micro-batch loop with DDP `no_sync`, the Builder wiring, the wandb
  `batch.micro_batch_size` field, and the 11 accumulation/validation tests.
  `_training_step` is back to one forward/backward per batch. Clipping and
  per-group LR logging remain.
- `tests/unit/core/test_trainer_accumulation.py` was renamed to
  `test_trainer_clipping_and_lr.py`.
- The DDP test now compares 2 ranks × batch 2 against 1 process × batch 4.
- Configs: `batch_size` is 4 per rank → global batch 32 × 4096 =
  131,072 tokens. Smoke: 240 updates, eval every 120. Sweep base: 22,888
  updates, eval every 2,288 (10%), warmup 22 updates.
- Not measured: throughput at batch 4 per GPU. Smoke job 773184 used 16 per
  rank (micro 8).
- Full suite: `376 passed in 125.89s`.

## Pool Job

- Job 773185 failed: `pyo3_runtime.PanicException: OS can't spawn worker
  thread: Resource temporarily unavailable`. 96 processes each start an
  `hf_xet` thread pool, and `ulimit -u` is 2048 on compute nodes too. The job
  was cancelled.
- The sbatch now sets `HF_HUB_DISABLE_XET=1` and limits
  OMP/MKL/RAYON/ARROW_IO threads.
- Job 773200 is running on `25a-hgpn078`: 98 processes, 315 threads.

## Not Verified

- **GPU.** No Slurm job was submitted, so memory at `micro_batch_size=8`,
  NCCL, throughput, and online Flock logging from compute nodes are unknown.
  Whether a short `dev` job consumes a grant is an open question for the TA.
- **Full data pool run.** The CPU partitions reject ACD115198, so where to run
  ~96 workers is open.
- **Still missing:** checkpoint/resume, `torch.compile`, fused CE, async data
  loading, and the 200B endpoint with partial global batches (`RankShard`
  rejects them; 200e9/4096 = 5^11 samples).
- **Recipe choice.** `max_grad_norm` is `null` in the configs because clipping
  is not in the sweep plan.
