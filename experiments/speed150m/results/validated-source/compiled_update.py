"""Compile the constant-LR plateau without changing scheduled LR representation."""
import torch


def compile_update(optimizer, *, plateau_lrs=None, backend='inductor'):
    groups = optimizer.param_groups
    rates = list(plateau_lrs) if plateau_lrs is not None else [group['lr'] for group in groups]
    eager = optimizer.step
    compiled = torch.compile(eager, backend=backend)

    def step():
        if all(group['lr'] == rate for group, rate in zip(groups, rates, strict=True)):
            compiled()
        else:
            eager()

    return step
