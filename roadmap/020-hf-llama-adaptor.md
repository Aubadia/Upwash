# 020 — Hugging Face Llama Model Adaptor

## Status — 2026-11-13

Implemented `config_adaptor` with the architecture mapping below and corrected the factory
name to the agreed `build_llama_model` API.

Verification: `./test.sh tests/unit/test_adaptor.py tests/unit/test_model_config.py`
→ exit 0, `58 passed in 2.76s`. CPU construction and the 4,432-parameter count
are Green. The full suite was not rerun. No current blocker; next agree on tests
for initialization/output scaling and detailed forward behavior. Custom
initialization and output scaling remain unimplemented.

## Historical Baseline

The user selected Hugging Face Llama as the model implementation and requested
TDD for `src/upwash/adaptor.py`. That file is currently empty. Production code
remains owned by the user.

Verified baseline:

- `./test.sh tests/unit/test_model_config.py` → exit 0, `57 passed in 0.04s`.
- `./test.sh` → exit 0, `276 passed in 4.69s`.
- The local environment has torch 2.15.0 and no installed `transformers`.
  `pyproject.toml` and `uv.lock` do not currently declare transformers.

## Agreed First Increment

The user rejected returning a config and approved:

```python
def build_llama_model(config: ModelConfig) -> LlamaForCausalLM:
    ...
```

Return a randomly initialized HF Llama causal language model on CPU, without
downloading pretrained weights. The first test checks the concrete model type,
CPU placement of all parameters, and actual parameter count. Dimensions are
vocabulary 32, 2 layers, hidden width 16, MLP width 24, 4 query heads and 2 KV
heads. Expected parameters: `32*16 + 2*(512 + 256 + 1152 + 32) + 16 = 4432`.
This assumes tied embeddings and bias-free projections, as agreed in task 019.
The test preserves the caller's CPU RNG state.

## Verified Red — 2026-11-13

The user added transformers with `uv add transformers`. The exact locked and
installed version is **5.22.2**, with torch **2.15.0**. Use this version for
subsequent HF reference checks; the earlier 5.2.0 documentation link is only
historical design context.

Before rerunning the test, a standalone Python check successfully constructed
a real `transformers.LlamaForCausalLM` from `LlamaConfig` with the test's tiny
architecture, tied embeddings and bias-free projections. All parameters were
on CPU and the total was exactly 4,432. This confirms reference construction
and the independently calculated count, not initialization or forward parity.
The check preserved CPU RNG state and downloaded no pretrained weights.

- `./test.sh tests/unit/test_adaptor.py tests/unit/test_model_config.py`
  → exit 1, `1 failed, 57 passed in 2.56s`.
- `./test.sh` → exit 1, `1 failed, 276 passed in 3.50s`.
- Sole failure: `ImportError: cannot import name 'build_llama_model' from
  'upwash.adaptor'`. The file remains empty. This is the intended missing-API
  Red; the new model assertions have not executed on Upwash yet.
- Next: user implements `build_llama_model(config)` in `src/upwash/adaptor.py`,
  returning the configured CPU HF model, then rerun for Green. Initialization,
  logits scaling and detailed architecture/forward behavior remain later tests.
- The agent changed documentation only in this verification step. Dependency
  changes belong to the user; no production code was modified.

## Historical Verification — Environment Blocker

- Added `tests/unit/test_adaptor.py` with one small model-construction test.
- `./test.sh tests/unit/test_adaptor.py tests/unit/test_model_config.py`
  → exit 1, `1 failed, 57 passed in 1.21s`.
- Failure: `ModuleNotFoundError: No module named 'transformers'` at the test's
  HF import. Upwash's missing function and model assertions have not executed.
  This is an environment blocker, not the intended Red.
- No production code, dependencies, lockfile, or test configuration changed.
- Next: agree on and prepare an exact transformers dependency version, verify
  a real tiny HF model constructs successfully, then rerun for an Upwash-side
  Red and hand implementation to the user. No HF oracle was executed yet.

The remaining config mapping below is design context, not a separate agreed
public function or verified behavior.

## Architecture Mapping Context

Model construction needs the following architecture settings:

| Upwash | Hugging Face |
| --- | --- |
| `vocab_size` | `vocab_size` |
| `n_layers` | `num_hidden_layers` |
| `d_model` | `hidden_size` |
| `d_ff` | `intermediate_size` |
| `n_heads` | `num_attention_heads` |
| `n_kv_heads` | `num_key_value_heads` |
| `max_seq_len` | `max_position_embeddings` |
| `norm_eps` | `rms_norm_eps` |
| `head_dim` | `head_dim` |

Also preserve `rope_theta` using the selected reference version's RoPE API,
with no RoPE scaling, and explicitly select tied embeddings, SiLU, bias-free
attention/MLP projections, and zero attention dropout.

HF documentation consulted:
[LlamaConfig (5.2.0)](https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/llama).
This documentation version is not yet an agreed dependency pin. Select and
record an exact transformers version before writing HF-dependent tests; discuss
dependency changes before applying them. No dependency was installed or changed.

Config translation alone does not implement task 019's embedding std,
per-matrix hidden initialization, or tied-output logits multiplier. Model
construction and the existing Trainer's tensor-output calling convention need
separate increments. No model weights are downloaded in the proposed test.

The former config-returning API proposal was rejected. Initialization rules,
output scaling, detailed architecture mapping and forward/backward behavior
remain subsequent test increments; the first test only covers construction,
type, device, and parameter count.
