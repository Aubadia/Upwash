import math

import pytest
import torch

from upwash.config import ModelConfig

# Option-B initialization for the tied-embedding HF Llama (roadmap/019-model-config.md):
# the embedding E has std `embedding_std`, hidden projections have std 1/sqrt(d_in),
# norms are ones, and logits are alpha * h @ E^T with
# alpha = (output_std_coeff / d_model) / embedding_std. The multiplier is applied by
# a forward hook on lm_head, so the model stays a plain LlamaForCausalLM.

PROJECTIONS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def small_config(**overrides):
    # Wide enough that sample standard deviations are within a few percent.
    fields = {
        "vocab_size": 512,
        "n_layers": 2,
        "d_model": 64,
        "d_ff": 96,
        "n_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 16,
        "rope_theta": 500_000.0,
        "norm_eps": 1e-5,
        "embedding_std": 0.1,
        "output_std_coeff": 1.0,
    }
    return ModelConfig(**{**fields, **overrides})


def build(config, seed=0):
    from upwash.adaptor import build_llama_model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build_llama_model(config)


def unscaled_copy(model, config):
    """A plain LlamaForCausalLM with the same weights and no output multiplier."""
    from transformers import LlamaForCausalLM

    from upwash.adaptor import config_adaptor

    reference = LlamaForCausalLM(config_adaptor(config))
    reference.load_state_dict(model.state_dict())
    return reference


def random_tokens(config, shape=(4, 16), seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, config.vocab_size, shape, generator=generator)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"d_model": 768}, 10 / 768),  # 150M: (1/768) / 0.1
        ({"d_model": 1_792, "n_heads": 28, "n_kv_heads": 7}, 10 / 1_792),  # 750M and 1B
        ({"output_std_coeff": 2.0, "embedding_std": 0.05}, (2.0 / 64) / 0.05),
    ],
    ids=["150M", "1B", "coefficients"],
)
def test_output_multiplier_matches_the_option_b_formula(overrides, expected):
    from upwash.adaptor import output_multiplier

    assert output_multiplier(small_config(**overrides)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("embedding_std", [0.1, 0.05])
def test_embedding_is_initialized_with_the_configured_std(embedding_std):
    model = build(small_config(embedding_std=embedding_std))
    weight = model.model.embed_tokens.weight.detach()

    # 512 x 64 samples: the sample std is within ~0.4% of the true std.
    assert weight.std().item() == pytest.approx(embedding_std, rel=0.05)
    assert abs(weight.mean().item()) < 0.1 * embedding_std
    # Tied: initializing hidden layers must not overwrite E through lm_head.
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_hidden_projections_use_inverse_sqrt_fan_in(projection):
    model = build(small_config())

    for layer in model.model.layers:
        module = layer.get_submodule(projection)
        expected = module.in_features ** -0.5  # 0.125 for d_in=64, ~0.102 for d_in=96
        # At least 2,048 samples per matrix: the sample std is within ~2%.
        assert module.weight.detach().std().item() == pytest.approx(expected, rel=0.1), projection


def test_norm_weights_are_ones():
    model = build(small_config())

    norms = [layer.input_layernorm for layer in model.model.layers]
    norms += [layer.post_attention_layernorm for layer in model.model.layers]
    norms.append(model.model.norm)
    for norm in norms:
        torch.testing.assert_close(norm.weight.detach(), torch.ones(64), rtol=0, atol=0)


def test_logits_are_the_output_multiplier_times_the_unscaled_llama_logits():
    from transformers import LlamaForCausalLM

    from upwash.adaptor import output_multiplier

    config = small_config()
    model = build(config).eval()
    reference = unscaled_copy(model, config).eval()
    inputs = random_tokens(config)

    with torch.no_grad():
        logits = model(inputs, use_cache=False).logits
        expected = output_multiplier(config) * reference(inputs, use_cache=False).logits

    assert isinstance(model, LlamaForCausalLM)
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)


def test_output_multiplier_participates_in_backpropagation():
    from upwash.adaptor import causal_lm_loss, output_multiplier

    config = small_config()
    model = build(config)
    reference = unscaled_copy(model, config)
    inputs, targets = random_tokens(config, seed=1), random_tokens(config, seed=2)
    alpha = output_multiplier(config)

    causal_lm_loss(model(inputs, use_cache=False), targets).backward()
    reference_logits = alpha * reference(inputs, use_cache=False).logits
    torch.nn.functional.cross_entropy(
        reference_logits.reshape(-1, config.vocab_size), targets.reshape(-1)
    ).backward()

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, rtol=1e-5, atol=1e-7, msg=name)


def test_initial_loss_is_close_to_the_uniform_prediction_loss():
    from upwash.adaptor import causal_lm_loss

    config = small_config()
    model = build(config).eval()

    with torch.no_grad():
        loss = causal_lm_loss(model(random_tokens(config, seed=1), use_cache=False), random_tokens(config, seed=2))

    # Logit std is about alpha * sqrt(d) * embedding_std = 1/sqrt(64) = 0.125, so the
    # loss exceeds ln V by about 0.125**2 / 2 ~= 0.008. Without the multiplier the
    # logit std would be sqrt(64) * 0.1 = 0.8, about 0.3 above ln V.
    assert loss.item() == pytest.approx(math.log(config.vocab_size), abs=0.05)


def test_initialization_is_reproducible_from_the_seed():
    config = small_config()

    first, again, other = build(config, seed=0), build(config, seed=0), build(config, seed=1)

    for (name, a), b, c in zip(first.named_parameters(), again.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    assert not torch.equal(first.model.embed_tokens.weight, other.model.embed_tokens.weight)
