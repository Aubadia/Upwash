"""Inverse-depth residuals against a plain HF Llama with folded projections.

Reference: transformers 5.22.2 / torch 2.15.0, CPU float32. Independent copies
of weights and inputs; compare logits, parameter gradients, and saved export.
"""
from dataclasses import replace

import torch
from transformers import LlamaForCausalLM

from upwash.adaptor import build_llama_model, export_hf_model
from upwash.config import ModelConfig


def config():
    return ModelConfig(
        vocab_size=32, n_layers=3, d_model=16, d_ff=32, n_heads=4, n_kv_heads=2,
        max_seq_len=8, rope_theta=500000.0, norm_eps=1e-5,
        embedding_std=0.1, output_std_coeff=1.0,
    )


def test_inverse_depth_matches_folded_reference_logits_and_gradients(tmp_path):
    base = config()
    with torch.random.fork_rng():
        torch.manual_seed(7)
        reference = build_llama_model(base)
        initial = {k: v.detach().clone() for k, v in reference.state_dict().items()}
        with torch.no_grad():
            for layer in reference.model.layers:
                layer.self_attn.o_proj.weight.div_(base.n_layers)
                layer.mlp.down_proj.weight.div_(base.n_layers)
        inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        expected = reference(inputs.clone()).logits
        expected.square().mean().backward()  # Run the reference before Upwash.

        scaled_config = replace(base, residual_scale_by_depth=True)
        model = build_llama_model(scaled_config)
        model.load_state_dict(initial)
        observed = model(inputs.clone()).logits
        observed.square().mean().backward()
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-5)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected_grad = reference_parameters[name].grad
            if name.endswith(('self_attn.o_proj.weight', 'mlp.down_proj.weight')):
                expected_grad = expected_grad / base.n_layers
            torch.testing.assert_close(parameter.grad, expected_grad, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)

        export_hf_model(model, scaled_config).save_pretrained(tmp_path)
        reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
        torch.testing.assert_close(reloaded(inputs).logits, observed, atol=2e-6, rtol=2e-5)
        assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
