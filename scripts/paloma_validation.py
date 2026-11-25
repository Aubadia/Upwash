"""Next-token scoring for Upwash's fixed Paloma validation subset."""

import math

import torch

from upwash.adaptor import causal_lm_loss


def evaluate(model, tokenizer, validation, device, *, prefix_token_id, seq_len):
    loss_sum = 0.0
    tokens = 0
    nbytes = 0
    was_training = model.training
    model.eval()
    try:
        device = torch.device(device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda",
        ):
            for text in validation:
                ids = [prefix_token_id] + tokenizer.encode(text, add_special_tokens=False).ids
                nbytes += len(text.encode("utf-8"))
                for start in range(0, len(ids) - 1, seq_len):
                    window = torch.tensor(ids[start:start + seq_len + 1], device=device).unsqueeze(0)
                    targets = window[:, 1:]
                    loss_sum += causal_lm_loss(model(window[:, :-1]), targets).item() * targets.numel()
                    tokens += targets.numel()
    finally:
        model.train(was_training)
    if not tokens or not nbytes or not math.isfinite(loss_sum):
        raise RuntimeError("Paloma validation loss is unavailable or non-finite")
    return {
        "eval/loss": loss_sum / tokens,
        "eval/bpb": loss_sum / (math.log(2) * nbytes),
        "eval/tokens": tokens,
        "eval/bytes": nbytes,
    }
