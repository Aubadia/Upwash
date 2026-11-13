"""Gradient clipping, per-group LR logging, and data-parallel token counts."""

import pytest
import torch

from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer


def make_model():
    torch.manual_seed(0)
    return torch.nn.Linear(3, 2, bias=False)


def make_batches():
    torch.manual_seed(1)
    return [(torch.randn(6, 3), torch.randn(6, 2)) for _ in range(3)]


def train(model, *, max_grad_norm=None, lr=0.1, logger=None, **kwargs):
    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(), optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        max_grad_norm=max_grad_norm,
        logger=logger, logging_steps=1 if logger is not None else None, **kwargs,
    )
    trainer.train(make_batches())
    return model


def test_clipping_limits_the_update_and_logs_the_pre_clip_norm():
    logger = MemoryLogger()
    model = make_model()
    before = model.weight.detach().clone()
    inputs, targets = make_batches()[0]
    loss = torch.nn.MSELoss()(model(inputs), targets)
    loss.backward()
    expected_norm = model.weight.grad.norm().item()
    expected = before - 0.1 * model.weight.grad * (1e-3 / expected_norm)
    model.weight.grad = None

    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(), optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        max_grad_norm=1e-3, logger=logger, logging_steps=1,
    )
    trainer.train([(inputs, targets)])

    assert expected_norm > 1e-3
    assert logger.records[0]["metrics"]["train/grad_norm"] == pytest.approx(expected_norm, rel=1e-5)
    torch.testing.assert_close(model.weight.detach(), expected, rtol=1e-5, atol=1e-6)


def test_a_norm_below_the_limit_is_not_changed():
    reference = train(make_model())
    clipped = train(make_model(), max_grad_norm=1e6)
    torch.testing.assert_close(clipped.weight, reference.weight)


def test_each_parameter_group_learning_rate_is_logged():
    logger = MemoryLogger()
    model = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False), torch.nn.Linear(3, 2, bias=False))
    optimizer = torch.optim.SGD(
        [{"params": model[0].parameters(), "lr": 0.1}, {"params": model[1].parameters(), "lr": 0.25}],
    )
    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(), optimizer=optimizer, logger=logger, logging_steps=1,
    )
    trainer.train(make_batches()[:1])

    metrics = logger.records[0]["metrics"]
    lr_keys = {key for key in metrics if key.startswith("train/learning_rate")}
    assert lr_keys == {"train/learning_rate", "train/learning_rate/1"}  # no duplicate of group 0
    assert metrics["train/learning_rate"] == pytest.approx(0.1)
    assert metrics["train/learning_rate/1"] == pytest.approx(0.25)


def test_named_parameter_groups_label_their_learning_rate():
    logger = MemoryLogger()
    model = torch.nn.Sequential(torch.nn.Linear(3, 3, bias=False), torch.nn.Linear(3, 2, bias=False))
    optimizer = torch.optim.SGD([
        {"params": model[0].parameters(), "lr": 0.1, "name": "muon"},
        {"params": model[1].parameters(), "lr": 0.25, "name": "adamw"},
    ])
    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(), optimizer=optimizer, logger=logger, logging_steps=1,
    )
    trainer.train(make_batches()[:1])

    metrics = logger.records[0]["metrics"]
    assert {key for key in metrics if key.startswith("train/learning_rate")} == {
        "train/learning_rate", "train/learning_rate/adamw",
    }
    assert metrics["train/learning_rate/adamw"] == pytest.approx(0.25)


def test_single_group_logs_only_the_existing_learning_rate_key():
    logger = MemoryLogger()
    train(make_model(), logger=logger)
    assert not any(key.startswith("train/learning_rate/") for key in logger.records[0]["metrics"])


def test_token_metrics_count_every_data_parallel_replica():
    logger = MemoryLogger()
    train(make_model(), logger=logger, count_tokens=True, world_size=4)

    # Each rank's batch holds 6 x 3 input elements; four ranks process 72 per update.
    assert [r["metrics"]["train/total_tokens_seen"] for r in logger.records] == [72, 144, 216]
