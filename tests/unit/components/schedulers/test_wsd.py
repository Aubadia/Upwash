import math

import pytest
import torch

from upwash.components.optimizers.combined import CombinedOptimizer
from upwash.config import ScheduleConfig


@pytest.mark.parametrize(
    ("warmup_ratio", "decay_ratio", "shape", "power", "expected_factors"),
    [
        pytest.param(
            0.2, 0.3, "power", 1.0,
            [0.5, 1, 1, 1, 1, 1, 1, 2 / 3, 1 / 3, 0],
            id="ten-update-linear",
        ),
        pytest.param(
            0.25, 0.5, "cosine", 2.0,
            [0.5, 1, 1, 1, (2 + math.sqrt(2)) / 4, 0.5, (2 - math.sqrt(2)) / 4, 0],
            id="cosine-quarter-turns",
        ),
        pytest.param(
            0.25, 0.5, "power", 2.0,
            [0.5, 1, 1, 1, 9 / 16, 1 / 4, 1 / 16, 0],
            id="quadratic-decay",
        ),
        pytest.param(
            0.25, 0.5, "power", 1.5,
            [0.5, 1, 1, 1, 3 * math.sqrt(3) / 8, math.sqrt(2) / 4, 1 / 8, 0],
            id="fractional-power-decay",
        ),
        pytest.param(0.0, 0.5, "power", 1.0, [1, 1, 0.5, 0], id="no-warmup"),
        pytest.param(0.5, 0.0, "power", 1.0, [0.5, 1, 1, 1], id="no-decay"),
        pytest.param(0.0, 0.0, "power", 1.0, [1, 1, 1, 1], id="stable-only"),
        pytest.param(1.0, 0.0, "power", 1.0, [0.25, 0.5, 0.75, 1], id="warmup-only"),
        pytest.param(0.0, 1.0, "power", 1.0, [0.75, 0.5, 0.25, 0], id="decay-only"),
        pytest.param(0.5, 0.5, "power", 1.0, [0.5, 1, 0.5, 0], id="no-stable-phase"),
        pytest.param(
            0.4, 0.4, "power", 1.0, [0.5, 1, 1, 1, 1, 0.5, 0],
            id="fractional-phase-lengths-truncate",
        ),
        pytest.param(
            0.1, 0.1, "power", 1.0, [1, 1, 1, 1],
            id="sub-update-phases-truncate-to-zero",
        ),
        pytest.param(0.0, 1.0, "power", 1.0, [0], id="single-decay-update"),
    ],
)
def test_wsd_prepares_update_lrs_for_muon_and_adamw(
    warmup_ratio, decay_ratio, shape, power, expected_factors,
):
    muon = torch.optim.Muon(
        [torch.nn.Parameter(torch.zeros(2, 2))],
        lr=0.008,
        adjust_lr_fn="match_rms_adamw",
    )
    adam = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=0.0128)
    optimizer = CombinedOptimizer([muon, adam])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    config = ScheduleConfig(
        warmup_ratio=warmup_ratio,
        decay_ratio=decay_ratio,
        decay_shape=shape,
        decay_power=power,
    )

    from upwash.components.schedulers.wsd import build_wsd_scheduler

    total_steps = len(expected_factors)
    lr_scheduler = build_wsd_scheduler(optimizer, config=config, total_steps=total_steps)

    observed_lrs = [[], []]
    for update in range(total_steps):
        # Construction prepares update 1; step() prepares the next update.
        for index, group in enumerate(optimizer.param_groups):
            observed_lrs[index].append(group["lr"])
        if update < total_steps - 1:
            lr_scheduler.step()

    for actual, base_lr in zip(observed_lrs, base_lrs, strict=True):
        assert actual == pytest.approx(
            [base_lr * factor for factor in expected_factors], rel=1e-10, abs=1e-12,
        )
        if expected_factors[-1] == 0:
            assert actual[-1] == 0.0

    # Changes must reach the original optimizers, not copied group dictionaries.
    assert muon.param_groups[0]["lr"] == observed_lrs[0][-1]
    assert adam.param_groups[0]["lr"] == observed_lrs[1][-1]
