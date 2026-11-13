import math
from copy import deepcopy

import torch

from upwash.config import ModelConfig


def test_build_llama_model_returns_a_cpu_llama_with_the_configured_parameter_count():
    from transformers import LlamaForCausalLM

    from upwash.adaptor import build_llama_model

    config = ModelConfig(
        vocab_size=32,
        n_layers=2,
        d_model=16,
        d_ff=24,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        rope_theta=500_000.0,
        norm_eps=1e-5,
        embedding_std=0.1,
        output_std_coeff=1.0,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_llama_model(config)

    assert isinstance(model, LlamaForCausalLM)
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    # 32*16 + 2*(2*16**2 + 2*16*8 + 3*16*24 + 2*16) + 16.
    assert config.num_parameters() == 4_432
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_432


def test_causal_lm_loss_uses_all_already_aligned_targets_without_shifting():
    from transformers.modeling_outputs import CausalLMOutputWithPast

    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[0, 1, 0]])
    original_targets = targets.clone()
    outputs = CausalLMOutputWithPast(logits=logits)
    expected_loss = (
        math.log1p(math.exp(-2)) + math.log1p(math.exp(-1)) + math.log(2)
    ) / 3
    expected_grad = torch.tensor([[
        [-1 / (1 + math.exp(2)), 1 / (1 + math.exp(2))],
        [1 / (1 + math.e), -1 / (1 + math.e)],
        [-0.5, 0.5],
    ]]) / 3

    from upwash.adaptor import causal_lm_loss

    loss = causal_lm_loss(outputs, targets)

    assert loss.ndim == 0
    torch.testing.assert_close(loss, torch.tensor(expected_loss), rtol=1e-6, atol=1e-7)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected_grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)


def test_causal_lm_loss_matches_reference_loss_and_real_llama_gradients():
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
            attention_dropout=0.0,
        )).to(device="cpu", dtype=torch.float32)
    reference = deepcopy(model)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 7], [5, 6, 8]])

    # torch 2.15.0 reference executes first with independent model/input state.
    reference_outputs = reference(inputs.clone(), use_cache=False)
    expected_loss = torch.nn.functional.cross_entropy(
        reference_outputs.logits.transpose(1, 2), targets.clone(), reduction="mean",
    )
    expected_loss.backward()
    expected_parameters = dict(reference.named_parameters())
    assert reference.model.embed_tokens.weight.grad is not None
    assert torch.count_nonzero(reference.model.embed_tokens.weight.grad).item() > 0

    from upwash.adaptor import causal_lm_loss

    outputs = model(inputs.clone(), use_cache=False)
    actual_loss = causal_lm_loss(outputs, targets.clone())

    assert actual_loss.ndim == 0
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    actual_loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert expected_parameters[name].grad is not None, name
        torch.testing.assert_close(
            parameter.grad, expected_parameters[name].grad, rtol=1e-5, atol=1e-7,
        )
