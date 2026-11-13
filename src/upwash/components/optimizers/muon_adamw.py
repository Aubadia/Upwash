import torch

from upwash.config import ModelConfig, OptimizerConfig
from upwash.components.optimizers.combined import CombinedOptimizer


def build_muon_adamw(
    model: torch.nn.Module,
    *,
    config: OptimizerConfig,
    model_config: ModelConfig,
) -> CombinedOptimizer:
    muon_parameters = []
    adam_parameters = []
    # `named_parameters()` deduplicates tied weights
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_embedding = "embed_tokens" in name or "lm_head" in name
        if parameter.ndim == 2 and not is_embedding:
            muon_parameters.append(parameter)
        else:
            adam_parameters.append(parameter)

    if not muon_parameters and not adam_parameters:
        raise ValueError("model has no trainable parameters")

    weight_decay = config.weight_decay / model_config.d_model
    optimizers = []
    if muon_parameters:
        optimizers.append(
            torch.optim.Muon(
                muon_parameters,
                lr=config.lr,
                momentum=config.muon_beta,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        )
    if adam_parameters:
        optimizers.append(
            torch.optim.AdamW(
                adam_parameters,
                lr=config.lr * config.adam_lr_multiplier,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=weight_decay,  # TODO: Consider adapt seperate weight decay
            )
        )
        
    return CombinedOptimizer(optimizers)
