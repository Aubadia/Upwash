# 019 — Model Configuration

## Behavior

This task describes a Llama-3.2-style model in configuration, including the initialization settings the spec requires us to record. It also adds two derived values: `head_dim` and the parameter count. The parameter count gives the 200 tokens-per-parameter budget. The model itself is not part of this task.

Branch `feat/optimizer-schedule-config`, as the third commit after task 017 (`7a8ed2c`) and task 018 (`2659838`), in the same pull request. The pull request opens after this task. `TrainingConfig` run settings and the top-level experiment config come in a later pull request, after the `logging_steps` meaning is agreed with the teammate working on logging.

## Spec Requirements Used

- Architecture: causal decoder blocks, RMSNorm, SwiGLU, RoPE, grouped-query attention, and **tied input and output embeddings**. These are listed as required features. Model dimensions, head dimensions, query-to-KV ratios, and attention scaling may vary.
- Head dimension: `d_h = d / h_q`. `d` must be divisible by `h_q`, `h_q` by `h_kv`, and standard RoPE needs an even head dimension.
- Parameter counts assume vocabulary 128,256, tied embeddings, bias-free projections, and two RMSNorms per layer plus a final RMSNorm.
- Logging: the run config must include the model configuration and its initialization.

The reference table lists 150M with 12 query heads and 36 KV heads. The agent recomputed every row with the spec's counting rules:

```text
V·d + L·(2d² + 2·d·(h_kv·d_h) + 3·d·m + 2d) + d
```

All five rows match exactly when 150M uses **3** KV heads (ratio 4:1, like the other rows). 36 is not valid (12 isn't divisible by 36) and gives 169,289,472 parameters. The tests use 3.

## Initialization Decision — 2026-11-13 (option B)

The experiment plan's rules were embedding std 0.1 and LM head std `1/d`. With tied embeddings they share one matrix, so they can't be initialized separately. Agreed approach, open to change after experiments:

- `E` is initialized with std `embedding_std` (0.1).
- Logits are `α · h @ Eᵀ` with `α = (output_std_coeff / d_model) / embedding_std`. At initialization this matches an LM head with std `output_std_coeff / d_model` (`1/d`), so the first logits are near zero and the loss is near `ln(128256) ≈ 11.76`.
- With the planned values, `α = 10/d`: 0.0130 at d=768, 0.0098 at 1,024, 0.0065 at 1,536, and 0.0056 at 1,792.
- Hidden matrices use std `1/sqrt(d_in)`. That rule isn't swept, so it is fixed in the model code rather than stored as a field.

Open: the teammates should confirm this was the intent of "LM head 1/d". An optional 150M comparison against standard tied init (std 0.02, no multiplier) could support the choice in the research log.

## Agreed Interface

```python
from upwash.config import ModelConfig

config = ModelConfig(
    vocab_size=128_256,
    n_layers=6,
    d_model=768,
    d_ff=3_072,            # MLP width m
    n_heads=12,            # query heads
    n_kv_heads=3,
    max_seq_len=4096,
    rope_theta=500_000.0,  # Llama 3
    norm_eps=1e-5,
    embedding_std=0.1,
    output_std_coeff=1.0,  # effective output std = output_std_coeff / d_model
)
config.head_dim           # d_model // n_heads
config.num_parameters()   # spec counting rules
```

The rules follow tasks 017 and 018:

- It is a dataclass, every field is required, and invalid values raise `ValueError`. A missing field raises `TypeError` naming it. `to_dict()` returns every field and doesn't include `head_dim`.
- Sizes (`vocab_size`, `n_layers`, `d_model`, `d_ff`, `n_heads`, `n_kv_heads`, `max_seq_len`) must be positive. They are checked before the divisibility rules.
- `d_model % n_heads == 0`, `n_heads % n_kv_heads == 0`, and an even `head_dim`.
- `rope_theta`, `norm_eps`, `embedding_std`, and `output_std_coeff` must be finite and greater than 0.

Fixed in model code, not fields: tied embeddings, bias-free projections, hidden init `1/sqrt(d_in)`, attention scale `1/sqrt(d_h)`, and no RoPE scaling.

Deferred:

- Checking that `max_seq_len` equals `DataConfig.seq_len`, in the top-level experiment config.
- An attention-scale field, if attention scaling is explored.
- Non-integer input types.

## Tests

`tests/unit/test_model_config.py`:

- All fields are exposed. Each missing field raises `TypeError` naming it.
- `num_parameters()` equals the spec's actual parameter count for 150M, 300M, 500M, 750M, and 1B. This uses the spec table as the oracle, with 150M at 3 KV heads.
- `head_dim` is 64 for 150M, 500M, and 1B.
- 12 KV heads (multi-head) and 1 KV head (multi-query) are accepted.
- 0 and -1 are rejected for each size field.
- These are rejected: `d_model=770` (not divisible by 12), `n_kv_heads` 5 and 36 (12 isn't divisible by either), and `d_model=780` (odd head dim 65).
- 0, -1, NaN, and infinity are rejected for each of the four scalars.
- `to_dict()` returns every field.

## Verified Red — 2026-11-13

- Command (Docker, `upwash-venv-pr5`): `PYTHONPATH=src uv run pytest -q tests/unit/test_model_config.py`.
- Result: collection error, `ImportError: cannot import name 'ModelConfig' from 'upwash.config'`. Missing API only.
- Rest of the suite, ignoring the file: `188 passed`.
- Test sanity check: a scratch reference implementation outside the repository passed the file (`57 passed`). It was loaded as a pytest plugin.

## Verified Green — 2026-11-13

The user implemented `ModelConfig`:

- It is a dataclass. `__post_init__` checks, in order: the seven sizes are positive (loop), `d_model % n_heads`, `n_heads % n_kv_heads`, an even `head_dim`, and the four scalars are finite and positive (loop).
- `head_dim` is a property.
- `num_parameters()` sums `2d² + 2·d·kv_dim + 3·d·d_ff + 2d` per layer, plus `vocab_size·d` and the final norm.

Command (Docker, `upwash-venv-pr5`): `uv run pytest -q tests/unit/test_model_config.py` returned `57 passed`. Full suite: `245 passed`.

Red iterations during implementation:

- The first version returned `39 failed, 18 passed`, from three defects:
  - The method was named `__post_int__`, so the dataclass never called it, and 34 cases gave `DID NOT RAISE ValueError`.
  - Both validation loops had their `if` outside the loop body, so only the last field was checked.
  - The scalar check lacked `> 0`.
- At the user's explicit request, the agent fixed those three defects in `config.py` and nothing else. The result was `5 failed, 52 passed`, because `num_parameters` didn't exist yet.
- `num_parameters()` then used `d.pow(2)`, which fails because a Python `int` has no `pow` method (`AttributeError`, five cases). The user changed it to `d**2`.

Open, cosmetic: the divisibility messages say "divied", and the odd-`head_dim` check raises `ValueError` without a message.

Status: complete. Next: commit as the third commit, then open the pull request for 017–019.
