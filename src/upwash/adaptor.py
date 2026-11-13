import functools

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from upwash.config import ModelConfig

def causal_lm_loss(outputs, targets):
    logits = outputs.logits
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

def config_adaptor(config: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=config.vocab_size,
        num_hidden_layers=config.n_layers,
        hidden_size=config.d_model,
        intermediate_size=config.d_ff,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        max_position_embeddings=config.max_seq_len,
        rms_norm_eps=config.norm_eps,
        head_dim=config.head_dim,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        tie_word_embeddings=True,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )

def output_multiplier(config: ModelConfig) -> float:
    return config.output_std_coeff / config.d_model / config.embedding_std

def _scale_output(module, inputs, output, *, alpha):
    return output * alpha

def build_llama_model(config: ModelConfig) -> LlamaForCausalLM:
    model = LlamaForCausalLM(config_adaptor(config))

    with torch.no_grad():
        model.model.embed_tokens.weight.normal_(0.0, config.embedding_std)
        for layer in model.model.layers:
            for module in layer.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.normal_(0.0, module.in_features ** -0.5)

    hook = functools.partial(_scale_output, alpha=output_multiplier(config))
    model.lm_head.register_forward_hook(hook)
    return model
