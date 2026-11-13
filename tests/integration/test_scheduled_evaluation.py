import time

import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import TrainingConfig
from upwash.core.logger import MemoryLogger


@pytest.mark.parametrize(
    ("num_steps", "expected_steps", "expected_train_losses", "expected_eval_losses"),
    [
        (2, [2], [3.28], [1.7850666667]),
        (3, [2, 3], [3.28, 1.6384], [1.7850666667, 1.3659093333]),
    ],
    ids=["exact-interval", "final-partial-interval"],
)
def test_scheduled_evaluation_merges_loss_with_training_events_without_duplicates(
    num_steps, expected_steps, expected_train_losses, expected_eval_losses, make_optimizer_config,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=make_optimizer_config(lr=0.1), logging_steps=2, eval_steps=2,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )
    inputs = torch.ones((1, 1), dtype=torch.float32)
    targets = torch.full((1, 1), 2.0, dtype=torch.float32)
    # A reusable dev iterable with unequal batch sizes verifies target weighting.
    dev_batches = [
        (torch.ones((2, 1)), torch.ones((2, 1))),
        (torch.ones((1, 1)), torch.full((1, 1), 3.0)),
    ]
    evaluation_grad_modes = []

    def observe_eval(module, args):
        if not module.training:
            evaluation_grad_modes.append(torch.is_grad_enabled())

    handle = model.register_forward_pre_hook(observe_eval)
    try:
        state = trainer.train(
            iter([(inputs.clone(), targets.clone()) for _ in range(num_steps)]),
            eval_batches=dev_batches,
        )
    finally:
        handle.remove()

    assert state.global_step == num_steps
    torch.testing.assert_close(
        model.weight, torch.tensor([[2 * (1 - 0.8 ** num_steps)]]),
        rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        model.weight.grad, torch.tensor([[-4 * 0.8 ** (num_steps - 1)]]),
        rtol=1e-6, atol=1e-6,
    )
    assert model.training is True
    # Two dev batches per checkpoint; no duplicate evaluation at the final step.
    assert evaluation_grad_modes == [False] * (2 * len(expected_steps))
    assert [record["step"] for record in logger.records] == expected_steps
    metrics = [record["metrics"] for record in logger.records]
    assert [event["train/loss"] for event in metrics] == pytest.approx(
        expected_train_losses, rel=1e-6, abs=1e-6,
    )
    eval_losses = [event["eval/loss"] for event in metrics]
    assert all(isinstance(loss, float) for loss in eval_losses)
    # Post-update w=0.72/0.976: [2*(w-1)^2 + (w-3)^2] / 3.
    assert eval_losses == pytest.approx(expected_eval_losses, rel=1e-6, abs=1e-6)
    assert all("eval/perplexity" not in event for event in metrics)


@pytest.mark.parametrize("num_steps", [3, 4], ids=["final-log-boundary", "final-eval-boundary"])
def test_different_eval_intervals_exclude_evaluation_and_logging_from_throughput(
    monkeypatch, num_steps, make_optimizer_config,
):
    elapsed = [0.0]

    def advance(seconds):
        elapsed[0] += seconds

    class SlowLogger(MemoryLogger):
        def log(self, metrics, *, step):
            super().log(metrics, step=step)
            advance(50.0)

    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = SlowLogger()
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=make_optimizer_config(lr=0.1), logging_steps=3, eval_steps=2,
        ),
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
        count_tokens=True,
    )
    model.register_forward_pre_hook(
        lambda module, args: advance(2.0 if module.training else 100.0)
    )
    model.weight.register_hook(lambda grad: advance(1.0))
    trainer.optimizer.register_step_post_hook(
        lambda optimizer, args, kwargs: advance(1.0)
    )
    inputs = torch.ones(4, dtype=torch.int64)
    targets = torch.zeros(4, dtype=torch.int64)

    def batches():
        for _ in range(num_steps):
            advance(1.0)
            yield inputs.clone(), targets.clone()
        advance(20.0)

    monkeypatch.setattr(time, "perf_counter", lambda: elapsed[0])
    state = trainer.train(batches(), eval_batches=[(inputs.clone(), targets.clone())])

    assert state.global_step == num_steps
    assert model.training
    assert model.weight[1, 0].item() > 0
    assert [record["step"] for record in logger.records] == list(range(2, num_steps + 1))
    evaluation_events = [
        record for record in logger.records if "eval/loss" in record["metrics"]
    ]
    assert [event["step"] for event in evaluation_events] == [2, num_steps]
    train_events = [
        record for record in logger.records if "train/loss" in record["metrics"]
    ]
    assert [event["step"] for event in train_events] == (
        [3] if num_steps == 3 else [3, 4]
    )
    # Four training tokens / (1s loading + 2s forward + 1s backward + 1s SGD).
    assert [
        event["metrics"]["train/tokens_per_second"] for event in train_events
    ] == pytest.approx([0.8] * len(train_events), rel=1e-6, abs=1e-6)
    assert train_events[-1]["metrics"]["train/total_tokens_seen"] == 4 * num_steps
    assert "eval/loss" in logger.records[-1]["metrics"]


@pytest.mark.parametrize("logging_steps", [None, 2], ids=["no-train-events", "no-logger"])
def test_scheduled_evaluation_runs_independently_of_training_logging(logging_steps, make_optimizer_config):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger() if logging_steps is None else None
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=make_optimizer_config(lr=0.1), logging_steps=logging_steps, eval_steps=2,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )
    evaluated_weights = []

    def observe_eval(module, args):
        if not module.training:
            evaluated_weights.append(module.weight.detach().item())

    handle = model.register_forward_pre_hook(observe_eval)
    try:
        trainer.train(
            [(torch.ones((1, 1)), torch.full((1, 1), 2.0))] * 3,
            eval_batches=[(torch.ones((1, 1)), torch.ones((1, 1)))],
        )
    finally:
        handle.remove()

    assert evaluated_weights == pytest.approx([0.72, 0.976], rel=1e-6, abs=1e-6)
    assert model.training
    if logger is not None:
        assert [record["step"] for record in logger.records] == [2, 3]
        assert all(set(record["metrics"]) == {"eval/loss"} for record in logger.records)
