# 025 — Option-B Initialization for the HF Llama

## Behavior

Apply the option-B initialization recorded in [019](019-model-config.md) to the HF Llama built by `build_llama_model` (PR #10, `src/upwash/adaptor.py`). Until now the model used HF's default initialization (std 0.02, no output multiplier), while `ModelConfig.embedding_std` and `output_std_coeff`, which are logged in the run config, described something else.

Numbered 025 (renumbered from 024 on 2026-11-14, because PR #13's training entry point took 024). Branch `feat/llama-init`, from `main` at `be696b0` (PRs #10 and #11 merged).

## Agreed Interface — 2026-11-14 (option C)

```python
from upwash.adaptor import build_llama_model, output_multiplier

output_multiplier(config)          # alpha = (output_std_coeff / d_model) / embedding_std
model = build_llama_model(config)  # still a CPU LlamaForCausalLM
```

- `embed_tokens.weight` (tied with `lm_head.weight`) is normal with std `embedding_std`.
- Every `nn.Linear` inside `model.model.layers` (`q/k/v/o_proj`, `gate/up/down_proj`) is normal with std `in_features ** -0.5`. `lm_head` is not re-initialized, because it is the embedding.
- RMSNorm weights are ones, which is HF's default.
- Logits are multiplied by α through a forward hook on `lm_head`. The model type, `causal_lm_loss`, `config_adaptor`, and the entry script's calls are unchanged.

Why option C. These alternatives were considered on 2026-11-14:

- A: α in the loss. Any caller that reads logits without our loss gets unscaled logits silently. This includes Flock in-memory scoring (`ScoringModel`), `generate`, and HF-computed losses.
- B: a wrapper module returning a logits tensor. It is the cleanest long-term contract, but it changes the model and loss interfaces now, prefixes `state_dict` keys, and isn't an HF causal LM for Flock in-memory scoring.
- D: subclass `LlamaForCausalLM` and override `forward`. This couples to HF's forward signature and return types. An HF-computed loss would also be computed before scaling, and Flock can't load the custom class name.
- C keeps the model a standard HF Llama, so every consumer of logits sees α. The hook only touches `lm_head`, a plain `nn.Linear`, so it doesn't depend on HF's forward signature.

C's risks and mitigations:

- The hook is easy to miss. A test compares logits with an unscaled copy times α, and the docstring should state it.
- `state_dict` carries no α, which is also true for A and B. Build training checkpoints with `build_llama_model` before loading weights. Only an exported, α-folded model may be loaded with `from_pretrained`.
- A hook written as a `lambda` can't be pickled (`torch.save(model)`). Prefer a module-level function with `functools.partial`. This is not tested.
- Folding α into the final norm while the hook is still attached would apply α². The export should build a fresh `LlamaForCausalLM` and `load_state_dict`, then fold α.

Deferred:

- `export_hf_model(model, config)`, which builds a fresh model, loads the weights, and multiplies the final norm weight by α, for `save_pretrained` and the Flock CLI. It comes later, when Flock scoring is set up.
- `max_position_embeddings` remains `max_seq_len` (4096), pending the host's answer on Flock's 8192 requirement.

When the hand-written Llama replaces the HF model, the hook goes away and α moves into its `forward`. The export must then rename parameters. The entry point, `output_multiplier`, the initialization rules, and most of these tests carry over. HF Llama plus this hook becomes the reference implementation for differential tests of the hand-written model.

## Tests

`tests/unit/test_llama_init.py`, CPU, with a tiny config (vocab 512, 2 layers, d 64, d_ff 96, 4 query heads, 2 KV heads), so sample standard deviations are within a few percent:

- `output_multiplier` is 10/768 for 150M and 10/1792 for 750M/1B, and `(2/64)/0.05` for non-default coefficients.
- The embedding std is `embedding_std` (0.1 and 0.05, within 5%), its mean is near 0, and `lm_head.weight is embed_tokens.weight`. A wrong std here would also catch `lm_head` being re-initialized over E.
- Each of the seven projections in each layer has std `in_features ** -0.5` within 10%: 0.125 for d_in 64, and about 0.102 for `down_proj`.
- All RMSNorm weights are exactly ones.
- Logits equal `output_multiplier(config)` times the logits of a plain `LlamaForCausalLM` with the same `state_dict`, and the model is still a `LlamaForCausalLM`.
- Gradients of `causal_lm_loss` match those of cross-entropy on α × the reference logits, for every parameter.
- The initial loss on random tokens is within 0.05 of `ln 512`. The expected excess is about 0.008. Without α it would be about 0.3.
- The same seed gives identical parameters, and another seed changes the embedding.

## Verified Red — 2026-11-14

Command (Docker, `upwash-venv-pr5` with `transformers==5.22.2`): `PYTHONPATH=src uv run --no-sync pytest -q tests/unit/test_llama_init.py`.

Result: `14 failed, 3 passed`.

- Five cases fail with `ImportError: cannot import name 'output_multiplier' from 'upwash.adaptor'`: three multiplier cases, the logits case, and the backpropagation case.
- The embedding std cases fail with `assert 0.0199… == 0.1 ± 0.005` (and `0.05 ± 0.0025`), from HF's default std 0.02.
- The seven projection cases fail similarly, for example `q_proj`: `0.0202… == 0.125 ± 0.0125`.
- Three cases pass already, because they are invariants that HF's default also satisfies:
  - Norm weights are ones.
  - The initial loss is close to ln V: with std 0.02 and no α, the logit std is about 0.16.
  - The seed makes initialization reproducible.

All failures come from the missing behavior or API. No environment failure occurred.

Rest of the suite: `324 passed`.

Test sanity check: a scratch reference implementation outside the repository passed the full suite (`341 passed`). It used `normal_` on the embedding and on the Linear modules in `model.model.layers`, and `register_forward_hook(functools.partial(_scale_output, alpha=...))` on `lm_head`.

## Verified Green — 2026-11-14

The user implemented these in `src/upwash/adaptor.py`:

- `output_multiplier(config)` returns `config.output_std_coeff / config.d_model / config.embedding_std`, which evaluates left to right.
- `build_llama_model` builds `LlamaForCausalLM(config_adaptor(config))`, then initializes under `torch.no_grad()`:
  - `embed_tokens.weight.normal_(0, embedding_std)`.
  - Every `nn.Linear` in each `layer.modules()` gets `normal_(0, in_features ** -0.5)`.
- It registers `functools.partial(_scale_output, alpha=output_multiplier(config))` as a forward hook on `lm_head`.

Command (Docker, `upwash-venv-pr5`): `uv run --no-sync pytest -q` returned `341 passed`.

Red iterations during implementation, first version `19 failed, 322 passed`:

- `model.model.embeded_tokens` raised `AttributeError`. The teammate's adaptor and optimizer tests that build the model failed too.
- Not yet reached at that point:
  - `layers.module()` instead of `modules()`.
  - `module.infeatures` instead of `in_features`.
  - The hook was registered without `alpha`.
- `output_multiplier` computed α without `return`: `assert None == 0.0130…`.

Follow-up cleanup by the user, rechecked with `341 passed`:

- Imports are ordered standard library, then third party, then `upwash`.
- The helpers are defined before `build_llama_model`.
- The hook is built as a named `hook` variable before registration.

Cosmetic, open:

- `build_llama_model` has no docstring mentioning the `lm_head` hook.
- `git diff --check` reports a blank line at the end of `adaptor.py`.

Status: complete in code and tests, not yet committed. Next: commit and open a PR. Export with α folding comes with Flock scoring.
