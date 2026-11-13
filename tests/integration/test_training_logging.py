import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import OptimizerConfig, TrainingConfig
from upwash.core.logger import MemoryLogger


@pytest.mark.parametrize(
    ("num_steps", "expected_steps", "expected_losses"),
    [
        (2, [2], [3.28]),
        (3, [2, 3], [3.28, 1.6384]),
        (4, [2, 4], [3.28, 1.343488]),
    ],
    ids=["exact-interval", "partial-interval", "two-exact-intervals"],
)
def test_built_trainer_logs_interval_mean_loss_and_the_final_partial_interval(
    num_steps, expected_steps, expected_losses
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    config = TrainingConfig(
        optimizer=OptimizerConfig(lr=0.1),
        logging_steps=2,
    )
    trainer = build_trainer(
        config=config,
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)
    batches = iter([(inputs.clone(), targets.clone()) for _ in range(num_steps)])

    state = trainer.train(batches)

    assert state.global_step == num_steps
    # Pre-update losses: 4, 2.56, 1.6384, 1.048576. Emit after completed updates.
    assert [record["step"] for record in logger.records] == expected_steps
    logged_losses = [record["metrics"]["train/loss"] for record in logger.records]
    assert all(isinstance(loss, float) for loss in logged_losses)
    assert logged_losses == pytest.approx(expected_losses, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("lr", "expected_weight"),
    [(0.1, 0.976), (0.05, 0.542)],
    ids=["lr-0.1", "lr-0.05"],
)
def test_built_trainer_logs_single_group_learning_rate_with_each_loss_event(
    lr, expected_weight
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=OptimizerConfig(lr=lr),
            logging_steps=2,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )
    inputs = torch.ones((1, 1), device="cpu", dtype=torch.float32)
    targets = torch.full((1, 1), 2.0, device="cpu", dtype=torch.float32)

    state = trainer.train([(inputs, targets)] * 3)

    assert state.global_step == 3
    torch.testing.assert_close(
        model.weight,
        torch.tensor([[expected_weight]], device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert [record["step"] for record in logger.records] == [2, 3]
    assert all("train/loss" in record["metrics"] for record in logger.records)
    rates = [record["metrics"]["train/learning_rate"] for record in logger.records]
    assert all(isinstance(rate, float) for rate in rates)
    assert rates == pytest.approx([lr, lr], rel=1e-6, abs=1e-8)


def test_built_trainer_logs_global_l2_gradient_norm_for_one_update():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, bias=True, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    logger = MemoryLogger()
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=OptimizerConfig(lr=0.1),
            logging_steps=1,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )
    inputs = torch.tensor([[3.0, 4.0]], device="cpu", dtype=torch.float32)
    targets = torch.ones((1, 1), device="cpu", dtype=torch.float32)

    state = trainer.train([(inputs, targets)])

    # At zero initialization, weight gradients are [-6, -8], bias gradient -2.
    # Global L2 is sqrt(36 + 64 + 4), rather than the sum of tensor norms (12).
    assert state.global_step == 1
    torch.testing.assert_close(model.weight, torch.tensor([[0.6, 0.8]]))
    torch.testing.assert_close(model.bias, torch.tensor([0.2]))
    torch.testing.assert_close(model.weight.grad, torch.tensor([[-6.0, -8.0]]))
    torch.testing.assert_close(model.bias.grad, torch.tensor([-2.0]))
    assert len(logger.records) == 1
    record = logger.records[0]
    assert record["step"] == 1
    metrics = record["metrics"]
    assert metrics["train/loss"] == pytest.approx(1.0)
    assert metrics["train/learning_rate"] == pytest.approx(0.1)
    grad_norm = metrics["train/grad_norm"]
    assert isinstance(grad_norm, float)
    assert grad_norm == pytest.approx(104 ** 0.5, rel=1e-6, abs=1e-6)
