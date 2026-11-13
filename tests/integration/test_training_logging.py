import time

import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import TrainingConfig
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
    num_steps, expected_steps, expected_losses, expected_grad_norms, make_optimizer_config, sgd_optimizer_factory
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    config = TrainingConfig(
        optimizer=make_optimizer_config(lr=0.1),
        logging_steps=2,
    )
    trainer = build_trainer(
        config=config,
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
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
        and "train/tokens_per_second" not in record["metrics"]
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
    lr, expected_weight, make_optimizer_config, sgd_optimizer_factory
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=make_optimizer_config(lr=lr),
            logging_steps=2,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
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
    extra_parameter_kind, make_optimizer_config, sgd_optimizer_factory,
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
            optimizer=make_optimizer_config(lr=0.1),
            logging_steps=1,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
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
    count_tokens, pad_token_id, input_row, expected_counts, make_optimizer_config, sgd_optimizer_factory,
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
            optimizer=make_optimizer_config(lr=0.1), logging_steps=2,
        ),
        model=model,
        loss_fn=next_token_loss,
        optimizer_factory=sgd_optimizer_factory,
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
            and "train/tokens_per_second" not in record["metrics"]
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


@pytest.mark.parametrize(
    ("pad_token_id", "input_pattern", "expected_totals", "expected_rates"),
    [
        (None, [1, 1], [16, 20], [1.6, 0.8]),
        (0, [0, 1], [8, 10], [0.8, 0.4]),
        (0, [0, 0], [0, 0], [0.0, 0.0]),
    ],
    ids=["no-padding", "mixed-padding", "all-padding"],
)
def test_throughput_includes_data_loading_and_excludes_logging_time(
    monkeypatch, pad_token_id, input_pattern, expected_totals, expected_rates,
):
    elapsed = [0.0]

    def advance(seconds):
        elapsed[0] += seconds

    class SlowLogger(MemoryLogger):
        def log(self, metrics, *, step):
            super().log(metrics, step=step)
            advance(100.0)

    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = SlowLogger()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        logger=logger,
        logging_steps=2,
        count_tokens=True,
        pad_token_id=pad_token_id,
    )
    # Advance simulated time without replacing real autograd or SGD updates.
    model.register_forward_pre_hook(lambda module, args: advance(2.0))
    model.weight.register_hook(lambda grad: advance(1.0))
    optimizer.register_step_post_hook(lambda optim, args, kwargs: advance(1.0))

    def batches():
        for num_tokens in (8, 8, 4):
            advance(1.0)
            yield (
                torch.tensor(input_pattern * (num_tokens // 2), dtype=torch.int64),
                torch.zeros(num_tokens, dtype=torch.int64, device="cpu"),
            )
        # Discovering exhaustion must not extend the last update's duration.
        advance(50.0)

    monkeypatch.setattr(time, "perf_counter", lambda: elapsed[0])
    state = trainer.train(batches())

    assert state.global_step == 3
    used_token = input_pattern[-1]
    assert model.weight[used_token, 0].item() > 0
    assert model.weight[used_token, 1].item() < 0
    assert [record["step"] for record in logger.records] == [2, 3]
    assert [
        record["metrics"]["train/total_tokens_seen"] for record in logger.records
    ] == expected_totals
    # Each batch costs 1s loading + 2s forward + 1s backward + 1s SGD.
    # Before padding exclusion, the final partial interval is 4 tokens / 5s.
    rates = [
        record["metrics"]["train/tokens_per_second"] for record in logger.records
    ]
    assert all(isinstance(rate, float) for rate in rates)
    assert rates == pytest.approx(expected_rates, rel=1e-6, abs=1e-6)

    # Neither an idle gap nor evaluation belongs to the next training interval.
    advance(500.0)
    trainer.evaluate([
        (torch.ones(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64))
    ])
    assert len(logger.records) == 2
    assert trainer.train(batches()).global_step == 3
    assert [record["step"] for record in logger.records] == [2, 3, 2, 3]
    assert [
        record["metrics"]["train/total_tokens_seen"] for record in logger.records[2:]
    ] == [total + expected_totals[-1] for total in expected_totals]
    assert [
        record["metrics"]["train/tokens_per_second"] for record in logger.records[2:]
    ] == pytest.approx(expected_rates, rel=1e-6, abs=1e-6)


def test_throughput_is_zero_when_clock_has_not_advanced(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2, device="cpu", dtype=torch.float32)
    logger = MemoryLogger()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        logger=logger,
        logging_steps=1,
        count_tokens=True,
    )
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)

    trainer.train([
        (torch.ones(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64))
    ])

    assert len(logger.records) == 1
    assert logger.records[0]["metrics"]["train/total_tokens_seen"] == 4
    rate = logger.records[0]["metrics"]["train/tokens_per_second"]
    assert isinstance(rate, float)
    assert rate == 0.0
