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
        # Training never reuses past key/values; a cache would only hold memory.
        use_cache=False,
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


def export_hf_model(model, config: ModelConfig, *, max_position_embeddings=None) -> LlamaForCausalLM:
    """Return a hook-free HF Llama whose logits equal the trained model's.

    The trained model applies alpha through a forward hook, which `save_pretrained`
    does not keep. With tied embeddings alpha cannot go into lm_head, so it is folded
    into the final RMSNorm weight, which feeds only lm_head:
    lm_head(alpha * norm(h)) == alpha * lm_head(norm(h)).
    A fresh model is built so the trained model keeps its hook and weights.
    """
    llama_config = config_adaptor(config)
    if max_position_embeddings is not None:
        llama_config.max_position_embeddings = max_position_embeddings
    exported = LlamaForCausalLM(llama_config)
    exported.load_state_dict(model.state_dict())
    with torch.no_grad():
        exported.model.norm.weight.mul_(output_multiplier(config))
    return exported.to(next(model.parameters()).device)
