"""Exporting the alpha-scaled training model as a plain HF Llama."""

import torch
from transformers import LlamaForCausalLM

from upwash.adaptor import build_llama_model, export_hf_model, output_multiplier
from upwash.config import ModelConfig

CONFIG = ModelConfig(
    vocab_size=64, n_layers=2, d_model=32, d_ff=64, n_heads=4, n_kv_heads=2,
    max_seq_len=16, rope_theta=500_000.0, norm_eps=1e-5, embedding_std=0.1, output_std_coeff=1.0,
)


def trained_model():
    torch.manual_seed(0)
    model = build_llama_model(CONFIG)
    with torch.no_grad():  # move away from initialization, including the final norm
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    return model.eval()


def tokens():
    return torch.randint(0, CONFIG.vocab_size, (2, CONFIG.max_seq_len), generator=torch.Generator().manual_seed(1))


def test_saved_and_reloaded_export_reproduces_the_training_logits(tmp_path):
    model = trained_model()
    with torch.no_grad():
        expected = model(tokens()).logits

    export_hf_model(model, CONFIG).save_pretrained(tmp_path)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        observed = reloaded(tokens()).logits

    assert output_multiplier(CONFIG) != 1.0
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
    assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr()


def test_export_leaves_the_training_model_unchanged():
    model = trained_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        expected = model(tokens()).logits

    export_hf_model(model, CONFIG)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
    with torch.no_grad():
        torch.testing.assert_close(model(tokens()).logits, expected)


def test_export_can_raise_max_position_embeddings_without_changing_logits(tmp_path):
    model = trained_model()
    with torch.no_grad():
        expected = model(tokens()).logits

    export_hf_model(model, CONFIG, max_position_embeddings=8192).save_pretrained(tmp_path)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path).eval()

    assert reloaded.config.max_position_embeddings == 8192
    with torch.no_grad():
        torch.testing.assert_close(reloaded(tokens()).logits, expected, rtol=1e-5, atol=1e-5)


def test_training_config_disables_the_kv_cache():
    assert build_llama_model(CONFIG).config.use_cache is False
