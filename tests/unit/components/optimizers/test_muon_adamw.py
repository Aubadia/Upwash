from collections import Counter
from copy import deepcopy

import torch

from upwash.adaptor import build_llama_model
from upwash.config import ModelConfig


def test_factory_partitions_tied_llama_parameters_and_applies_model_width(make_optimizer_config):
    model_config = ModelConfig(
        vocab_size=32,
        n_layers=1,
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
    config = make_optimizer_config(
        lr=0.008,
        muon_beta=0.95,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_lr_multiplier=1.6,
        weight_decay=0.16,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_llama_model(model_config)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}

    muon_names = {
        f"model.layers.0.self_attn.{projection}_proj.weight"
        for projection in ("q", "k", "v", "o")
    } | {
        f"model.layers.0.mlp.{projection}_proj.weight"
        for projection in ("gate", "up", "down")
    }
    adam_names = {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    }

    # Real torch 2.15.0 references run before the Upwash import. Independent
    # model copies prevent the oracle from sharing optimizer/model state.
    reference_parameters = dict(deepcopy(model).named_parameters())
    reference_muon = torch.optim.Muon(
        [reference_parameters[name] for name in sorted(muon_names)],
        lr=config.lr,
        momentum=config.muon_beta,
        weight_decay=config.weight_decay / model_config.d_model,
        adjust_lr_fn="match_rms_adamw",
    )
    reference_adam = torch.optim.AdamW(
        [reference_parameters[name] for name in sorted(adam_names)],
        lr=config.lr * config.adam_lr_multiplier,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay / model_config.d_model,
    )
    # Independently calculated width scaling, independent of the oracle.
    assert reference_muon.param_groups[0]["weight_decay"] == 0.01
    assert reference_adam.param_groups[0]["weight_decay"] == 0.01

    from upwash.components.optimizers.muon_adamw import build_muon_adamw

    optimizer = build_muon_adamw(model, config=config, model_config=model_config)

    names_by_id = {id(p): name for name, p in model.named_parameters()}
    assigned_names = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names_by_id[id(parameter)]
            assigned_names.append(name)
            if name in muon_names:
                expected = reference_muon.param_groups[0]
                keys = ("lr", "momentum", "weight_decay", "adjust_lr_fn")
            else:
                assert name in adam_names
                expected = reference_adam.param_groups[0]
                keys = ("lr", "betas", "weight_decay")
            for key in keys:
                assert group[key] == expected[key], (name, key)

    assert Counter(assigned_names) == Counter(muon_names | adam_names)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, initial_parameters[name], rtol=0, atol=0)
