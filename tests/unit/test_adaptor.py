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
