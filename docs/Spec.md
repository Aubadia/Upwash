# Predictable Scaling Lab Specification

This document consolidates the lab description supplied by the user on
2026-11-13. It records the supplied requirements, rather than a claim that
Upwash implements them. Implementation evidence belongs in the
[roadmap](../roadmap/README.md); component boundaries belong in
[Architecture.md](Architecture.md), and the incremental Trainer contract in
[Trainer.md](Trainer.md). Links below are supplied references; their current
contents and remote service behavior were not verified for this update.

## Goal and Required Runs

Develop a mostly handwritten, persistent training stack and use small-scale
experiments to select a recipe that transfers to a final 1B model. The final
objective is the lowest perplexity on the designated evaluation stack after
training from scratch on **exactly 200,000,000,000 non-padding FineWeb-Edu
tokens**, with a context window of **4096 tokens**.

Every occurrence of a training token counts, including repeated tokens on later
passes. Padding does not count. Faster execution does not permit extra tokens.
The checkpoint at the prescribed endpoint determines the final model score.

The required 750M calibration run uses the lab's scaling equation and trains
for **200 tokens per parameter**. The final 1B run has a fixed configuration
derived from that equation and the fixed 200B-token endpoint. Exploration may
use other sizes, intermediate scales, horizons, and recipes. Small-model
success alone is not evidence of transfer.

## Model Architecture

Required Llama-3.2-style features are causal decoder blocks, RMSNorm,
SwiGLU, rotary position embeddings (RoPE), grouped-query attention (GQA),
and tied input/output embeddings. Exploration may vary model dimensions,
head dimensions, query-to-KV ratios, and attention scaling.

Each layer has separate pre-attention and pre-MLP RMSNorm scale vectors:

```text
r = x + Attention(RMSNorm_attn(x))
y = r + MLP(RMSNorm_mlp(r))
MLP(z) = (SiLU(z W_gate) * (z W_up)) W_down
Q = RoPE(u W_Q), K = RoPE(u W_K), V = u W_V
A_i = softmax(s Q_i K_g(i)^T + causal_mask) V_g(i)
```

Concatenate query-head outputs and apply the output projection. Residuals
preserve `[B, T, d]`. Apply a final RMSNorm and project through the tied
embedding matrix to produce logits. The conventional attention scale is
`s = 1 / sqrt(d_h)`.

Under the reference dimension convention, `d_h = d / h_q` and
`d_KV = h_kv * d_h`. Require `d % h_q == 0`, `h_q % h_kv == 0`, and an even
head dimension for standard RoPE. Verify kernel support for the chosen head
dimension. MLP width `m` is independent of attention dimensions: `8d/3` and
`4d` are suggested starting ratios, optionally rounded up to multiples of
64, 128, or 256; these are heuristics, not mandatory dimensions.

The supplied reference table assumes vocabulary 128,256, tied embeddings,
bias-free projections, two RMSNorms per layer, and a final RMSNorm:

| Scale | Layers | Hidden d | MLP m | Query heads | KV heads in source | Parameters in source |
| --- | --- | --- | --- | --- | --- | --- |
| 150M | 6 | 768 | 3,072 | 12 | 36 | 149,825,280 |
| 300M | 9 | 1,024 | 5,184 | 16 | 4 | 298,273,792 |
| 500M | 11 | 1,536 | 4,672 | 24 | 6 | 498,731,520 |
| 750M | 13 | 1,792 | 5,888 | 28 | 7 | 745,749,760 |
| 1B | 15 | 1,792 | 8,000 | 28 | 7 | 995,432,704 |

For these assumptions, parameter counting is:

```text
P = V*d + L*(2*d^2 + 2*d*(h_kv*d_h) + 3*d*m + 2*d) + d
```

The source's 150M row is inconsistent: 36 KV heads violates the divisibility
rule and does not reproduce its parameter count. Upwash already uses **3 KV
heads**, which gives 149,825,280 parameters; see
[task 019 — Model Configuration](../roadmap/019-model-config.md). This is a
recorded project interpretation, not an edited quotation of the source table.

## Training Stack and Experiment Design

Maintain and understand the training logic, including model, optimizer, data
loading, packing, loss, and checkpointing. A Flock trainer fork is permitted.
PyTorch and existing attention-kernel/distributed-training libraries are
permitted. Both members of the pair must be able to explain promising methods
and why they should work.

Experiment candidates include optimizers, initialization, parameterization,
learning rates, schedules, and batch sizes. There is no prescribed peak LR;
its transfer depends on optimizer, parameterization, batch size, and horizon.
Recipes may change in response to evidence. Maintain a research log containing
hypotheses, the reason for each experiment, results, transfer limitations,
recipe changes, and supporting evidence, including unsuccessful experiments.

## Compute Grants

Grants belong to each pair. Each grant permits one run using at most its
allowance. GPU hours sum across devices: eight H200s for two hours consume
16 H200-hours. A smaller or early-stopped experiment still consumes one grant
at its assigned level; unused allowance cannot be split into replacement runs.
Resuming an interrupted run from its checkpoint remains part of the same run.

| H200-hours per run | Reference scale | Reference tokens | Grants in source | Availability in source |
| --- | --- | --- | --- | --- |
| 40 | 150M | 30B | approximately 60 | Throughout both weeks |
| 110 | 300M | 60B | 8 | Throughout both weeks |
| 266 | 500M | 100B | 3 | 1 before calibration, 4 additional after |
| 540 | 750M | 150B | 1 | Launch before 11/12 23:59 |

The allowances reference an optimized PyTorch trainer with `torch.compile`,
FlashAttention-3, packing, and 200 tokens per parameter; these describe the
allowance basis, not mandatory exploration configurations or methods.
One final 1B/200B run is allocated separately. The source specifies one final
training attempt per pair.

## Data and Evaluation

Exclude all **5,000 FineWeb-Edu evaluation samples** from training. Exclusion
must also be established for externally prepared token shards; successful
loading alone does not verify their provenance or absence of contamination.

The evaluation stack is:

| Task or suite | Supplied description |
| --- | --- |
| `fineweb_edu` | [cerulean-works/fineweb-edu-dev](https://huggingface.co/datasets/cerulean-works/fineweb-edu-dev) |
| `base_easy` / `base_easy:bpb` | OLMo 3 base easy suite |
| `paloma` | BPB across 309 domains, including domain breakdown |

Use [Flock evaluation](https://docs.cerulean.works/flock/main/guide/evals/run-evaluation/)
for scoring, with the supplied commands:

```bash
flock eval run --task fineweb_edu --model <repo@rev> --dtype float32 --out <dir>
flock eval run --suite base_easy --model <repo@rev> --dtype float32 --out <dir>
flock eval run --suite paloma --model <repo@rev> --dtype float32 --out <dir>
```

Evaluate at least once every 10% of intended training steps. For next-token
natural-log cross-entropy, `eval/loss` is the sum of scored token losses divided
by the number of scored tokens, and `eval/perplexity = exp(eval/loss)`.
Do not apply this transform to arbitrary regression losses or treat BPB as PPL.
The model/tokenizer export format and exact Flock compatibility must be
established separately from Upwash's local evaluator.

## Required Logging

Log training metrics under project **`lab6-predictable-scaling`** through the
wandb interface. Upwash's selected destination is Flock via `FlockLogger`;
adapter unit tests do not establish actual remote delivery.

| Metric | Lab requirement |
| --- | --- |
| `train/loss` | Average loss over `logging_steps`. |
| `train/grad_norm` | Gradient norm before clipping. |
| `train/learning_rate` | Learning rate; log parameter groups separately when rates differ. |
| `train/tokens_per_second` | Training throughput at the logging step. |
| `train/total_tokens_seen` | Cumulative non-padding training tokens, including repeats. |
| `eval/loss` | Token-averaged cross-entropy on the designated dev set. |
| `eval/perplexity` | Perplexity derived from that evaluation loss. |

The run configuration must include model configuration, initialization,
optimizer settings, batch size, schedule, and training duration. Distinguish
planned duration/token endpoint from observed progress and elapsed time.
The source does not prescribe detailed training-loss aggregation, gradient
norm sampling, group metric names, or throughput timing boundaries; Upwash's
incremental decisions and evidence live in
[task 019 — Training Loop Logging](../roadmap/019-training-loop-logging.md).

## Forecasting and Submission

**Before each required 750M calibration and final 1B run begins**, record and
submit a forecast for each benchmark in the experiment log containing:

1. Model/training configuration and intended token endpoint.
2. Estimated endpoint perplexity on the designated evaluation set.
3. An 80% prediction interval.
4. An explanation supported by experiments, baseline comparisons, scaling-law
   fits, or another stated estimation method.

The forecast applies to the submitted recipe. Never revise an original scored
forecast after seeing its outcome. During-run updates may be recorded
separately. Recipe changes between calibration and final training are allowed.
No fitting method or number of calibration experiments is prescribed.

The supplied forecast error is
`abs(ln(predicted_PPL) - ln(actual_PPL))`; lower is better, averaged across
the two designated runs. Intervals are recorded for discussion and do not
contribute to the score. Cross-benchmark aggregation needs clarification.

Before the final launch, commit the chosen recipe and submit its configuration
and forecast. After completion, upload weights and tokenizer to Hugging Face,
set `eval_model_id` in the OJ script, and include the training configuration,
run link, forecast submission, and research log so the recipe, token budget,
and history can be verified. Recording these requirements does not execute or
authorize remote submission, upload, or a compute run.

The main leaderboard uses final 1B/200B model perplexity; forecasting has a
separate leaderboard. Smaller-scale results are shared without competitive
PPL rankings or SOTA points. The source defers the final SOTA-point formula
until calibration results and allows adjustment based on research-log quality.

## Source Ambiguities and Project Boundaries

- **Scaling equation and fixed final configuration:** the source refers to a
  scaling equation without supplying it explicitly. The parameter-count
  formula above follows the stated architecture assumptions; do not assume it
  fully specifies the organizer's scaling rule. Confirm the official final
  configuration before launch.
- **500M grants:** the table says 3 grants, while availability says 1 plus 4
  additional grants. Preserve this discrepancy pending organizer clarification.
- **Calibration deadline:** the source uses both 11/12 23:59 and 11/12 24:00,
  without an explicit year or timezone. These are source labels, not a newly
  verified deadline or evidence that a run was launched on time.
- **Metric units:** the source describes perplexity-based evaluation/forecasts
  alongside BPB suites. Confirm each scored metric, its units, benchmark
  aggregation, and the applicable forecasting transform. Do not invent a
  conversion or leaderboard formula.
- **Exact endpoint:** current dataset construction truncates to
  `num_tokens // seq_len` complete samples. A configured budget or logged
  counter alone does not prove exact stopping, global distributed counting,
  or correct resume behavior. Agree on endpoint and final-update handling
  before their tests; 200B is divisible by 4096 but may not be divisible by a
  chosen global update's token count.
- **Implementation policy:** Muon/AdamW settings, WSD configuration, and the
  tied-embedding initialization/output multiplier in the model task are
  project choices, not mandatory lab recipes. Config validation does not prove
  runtime optimizer, scheduler, initialization, or model behavior.

## Verified Upwash Snapshot — 2026-11-14

Full suite (Docker, `upwash-venv-pr5`, `transformers==5.22.2`): **348 passed**, on branch `feat/llama-init` (task 025) after merging `main` at `412b09b` (PR #13). Since the 2026-11-13 snapshot below, these are implemented and tested on CPU:

- Device placement and bf16 mixed precision (task 020, `device-precision`).
- `eval/perplexity` with explicit opt-in (task 021, `eval-perplexity`), and the prepared fineweb-edu dev shard.
- HF Llama construction, Muon/AdamW, and WSD scheduling (PR #10).
- Option-B initialization with a forward-hook output multiplier (task 025, PR #12).
- A configured training entry point, `RunConfig` → `train(config)` (task 024, PR #13). Its Slurm CUDA smoke test is not yet run.

Still not established:

- Multi-group LR logging.
- Exact-budget stopping, checkpoint and resume, and real CUDA and distributed behavior.
- Remote metric delivery.
- Export with α folded into the final norm, and Flock scoring.
- Flock's 8192-context requirement against the 4096 training context.

The snapshot below is historical.

## Verified Upwash Snapshot — 2026-11-13

`./test.sh` returned exit 0: **276 passed in 4.67s** before this documentation
update. Source and relevant tests were inspected. This update adds no runtime
API, production implementation, test cases, dependencies, or configuration.

Existing coverage establishes configuration/parameter counting, packed-data
reading, real CPU gradient updates, interval logging/token counting/throughput,
and local manual/scheduled evaluation within their documented scope.
`eval/perplexity` is not implemented; its explicit LM opt-in API remains to be
agreed. Multi-group LR logging, actual Llama construction, runtime optimizer
and scheduler integration, exact-budget stopping, checkpoint/resume, GPU and
distributed behavior, dev exclusion for Flock shards, remote metric delivery,
and final export/scoring are not established by this passing suite.
