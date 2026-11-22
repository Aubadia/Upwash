import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import flash_attention_mask


def upwash_fa3_forward(
    model,
    query,
    key,
    value,
    attention_mask,
    *,
    scaling=None,
    dropout=0.0,
    **kwargs
):
    from flash_attn_interface import flash_attn_func

    device = query.device.type

    if torch.is_autocast_enabled(device):
        dtype = torch.get_autocast_dtype(device)
    else:
        dtype = model.q_proj.weight.dtype

    # llama qkv shape = (Batch, H_q, Seq, H_dim)
    # fa3 qkv shape = (Batch, Seq, H_q, H_dim)
    q = query.to(dtype).transpose(1, 2)
    k = key.to(dtype).transpose(1, 2)
    v = value.to(dtype).transpose(1, 2)

    if attention_mask is None:
        output = flash_attn_func(
            q, k, v,
            softmax_scale=scaling,
            causal=model.is_causal
        )
    else:
        raise ValueError("attention_mask and padding is not supported")

    return output, None

    # TODO: Handle padding


def register_upwash_fa3_wrapper():
    AttentionInterface.register(
        "upwash_fa3",
        upwash_fa3_forward
    )
    AttentionMaskInterface.register(
        "upwash_fa3",
        flash_attention_mask
    )
