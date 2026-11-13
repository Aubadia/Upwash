import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import OptimizerConfig, TrainingConfig
from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer


@pytest.mark.parametrize(
    ("num_steps", "expected_steps", "expected_losses", "expected_grad_norms"),
    [
        (2, [2], [3.28], [3.2]),
        (3, [2, 3], [3.28, 1.6384], [3.2, 2.56]),
        (4, [2, 4], [3.28, 1.343488], [3.2, 2.048]),
    ],
    ids=["exact-interval", "partial-interval", "two-exact-intervals"],
)
def test_built_trainer_logs_interval_mean_loss_and_the_final_partial_interval(
    num_steps, expected_steps, expected_losses, expected_grad_norms
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
    assert all(
        "train/total_tokens_seen" not in record["metrics"]
        for record in logger.records
    )
    # Pre-update gradients: -4, -3.2, -2.56, -2.048. Log the last step's L2.
    norms = [record["metrics"]["train/grad_norm"] for record in logger.records]
    assert all(isinstance(norm, float) for norm in norms)
    assert norms == pytest.approx(expected_grad_norms, rel=1e-6, abs=1e-6)
    torch.testing.assert_close(
        model.weight, torch.tensor([[2 * (1 - 0.8 ** num_steps)]]),
        rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        model.weight.grad, torch.tensor([[-4 * 0.8 ** (num_steps - 1)]]),
        rtol=1e-6, atol=1e-6,
    )


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


@pytest.mark.parametrize("extra_parameter_kind", ["none", "unused", "frozen"])
def test_built_trainer_logs_global_l2_gradient_norm_for_one_update(
    extra_parameter_kind,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, bias=True, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    if extra_parameter_kind != "none":
        model.register_parameter(
            "extra",
            torch.nn.Parameter(
                torch.tensor([100.0], device="cpu", dtype=torch.float32),
                requires_grad=extra_parameter_kind == "unused",
            ),
        )
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
    if extra_parameter_kind != "none":
        assert model.extra.grad is None
        torch.testing.assert_close(model.extra, torch.tensor([100.0]))
    assert len(logger.records) == 1
    record = logger.records[0]
    assert record["step"] == 1
    metrics = record["metrics"]
    assert metrics["train/loss"] == pytest.approx(1.0)
    assert metrics["train/learning_rate"] == pytest.approx(0.1)
    grad_norm = metrics["train/grad_norm"]
    assert isinstance(grad_norm, float)
    assert grad_norm == pytest.approx(104 ** 0.5, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize(
    ("count_tokens", "pad_token_id", "input_row", "expected_counts"),
    [
        (True, None, [0, 0, 0, 0], [16, 20]),
        (True, 0, [0, 1, 0, 1], [8, 10]),
        (True, 1, [0, 1, 0, 1], [8, 10]),
        (True, 0, [0, 0, 0, 0], [0, 0]),
        (False, 0, [0, 1, 0, 1], None),
    ],
    ids=["no-padding", "pad-zero", "pad-one", "all-padding", "disabled"],
)
def test_built_trainer_logs_total_input_tokens_including_final_partial_interval(
    count_tokens, pad_token_id, input_row, expected_counts,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()

    def next_token_loss(logits, targets):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2), targets.reshape(-1)
        )

    logger = MemoryLogger()
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=OptimizerConfig(lr=0.1), logging_steps=2,
        ),
        model=model,
        loss_fn=next_token_loss,
        optimizer_factory=torch.optim.SGD,
        logger=logger,
        count_tokens=count_tokens,
        pad_token_id=pad_token_id,
    )
    # Token id 0 is valid when padding is not configured. Count inputs only.
    batches = iter([
        (
            torch.tensor([input_row] * batch_size, dtype=torch.int64, device="cpu"),
            torch.ones((batch_size, 4), dtype=torch.int64, device="cpu"),
        )
        for batch_size in (2, 2, 1)
    ])

    state = trainer.train(batches)

    assert state.global_step == 3
    assert model.weight[0, 0].item() < 0
    assert model.weight[0, 1].item() > 0
    if 1 not in input_row:
        torch.testing.assert_close(model.weight[1], torch.zeros(2))
    assert [record["step"] for record in logger.records] == [2, 3]
    assert all("train/loss" in record["metrics"] for record in logger.records)
    if expected_counts is None:
        assert all(
            "train/total_tokens_seen" not in record["metrics"]
            for record in logger.records
        )
        return
    counts = [record["metrics"]["train/total_tokens_seen"] for record in logger.records]
    assert all(type(count) is int for count in counts)
    assert counts == expected_counts


def test_token_total_persists_across_training_calls_without_counting_evaluation():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        logger=logger,
        logging_steps=2,
        count_tokens=True,
        pad_token_id=0,
    )
    inputs = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    targets = torch.zeros(4, dtype=torch.int64)

    assert trainer.train([(inputs, targets)]).global_step == 1
    assert logger.records[0]["metrics"]["train/total_tokens_seen"] == 3
    trainer.evaluate([(inputs, targets)])
    assert len(logger.records) == 1
    # Counting continues even while metric emission is disabled.
    trainer.logger = None
    assert trainer.train([(inputs, targets)]).global_step == 1
    trainer.logger = logger
    assert trainer.train([(inputs, targets)]).global_step == 1

    assert [record["step"] for record in logger.records] == [1, 1]
    assert [
        record["metrics"]["train/total_tokens_seen"] for record in logger.records
    ] == [3, 9]
    assert model.weight[1, 0].item() > 0
