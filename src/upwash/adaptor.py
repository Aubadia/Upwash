from upwash.config import ModelConfig

from transformers import LlamaConfig, LlamaForCausalLM

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

def build_llama_model(config: ModelConfig) -> LlamaForCausalLM:
    llama_config = config_adaptor(config)
    return LlamaForCausalLM(llama_config)
