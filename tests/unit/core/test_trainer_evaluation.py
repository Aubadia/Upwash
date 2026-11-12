import pytest
import torch

from upwash.core.trainer import Trainer


def test_evaluate_weights_mean_loss_by_target_count_with_a_smaller_final_batch():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    # Two target positions per sample: 4 targets in the first batch, 2 in the last.
    # With zero predictions their mean squared losses are 1 and 9, respectively.
    batches = iter(
        [
            (torch.ones((2, 2, 1)), torch.ones((2, 2, 1))),
            (torch.ones((1, 2, 1)), torch.full((1, 2, 1), 3.0)),
        ]
    )

    result = trainer.evaluate(batches)

    assert isinstance(result["loss"], float)
    # Target-weighted mean: (4 * 1 + 2 * 9) / 6 = 11/3; batch mean would be 5.
    assert result["loss"] == pytest.approx(11 / 3, rel=1e-6, abs=1e-6)
    assert isinstance(result["num_targets"], int)
    assert result["num_targets"] == 6


@pytest.mark.parametrize("previous_training", [True, False], ids=["train", "eval"])
def test_evaluate_disables_gradients_and_restores_the_previous_mode(previous_training):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    model.train(previous_training)
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    observations = []

    def observe_forward(module, inputs, output):
        observations.append(
            (module.training, torch.is_grad_enabled(), output.requires_grad)
        )

    handle = model.register_forward_hook(observe_forward)
    try:
        # Explicitly enable gradients so the test cannot pass from an outer no-grad.
        with torch.enable_grad():
            trainer.evaluate([(torch.ones((1, 1)), torch.zeros((1, 1)))])
            assert torch.is_grad_enabled()
    finally:
        handle.remove()

    assert observations
    assert all(observation == (False, False, False) for observation in observations)
    assert model.training is previous_training


def test_evaluate_restores_training_mode_when_loss_computation_raises():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    model.train()
    error = RuntimeError("evaluation loss failed")

    def failing_loss(predictions, targets):
        raise error

    trainer = Trainer(
        model=model,
        loss_fn=failing_loss,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    with torch.enable_grad():
        with pytest.raises(RuntimeError) as raised:
            trainer.evaluate([(torch.ones((1, 1)), torch.zeros((1, 1)))])
        assert torch.is_grad_enabled()

    assert raised.value is error
    assert model.training is True


@pytest.mark.parametrize("as_iterator", [False, True], ids=["list", "iterator"])
def test_evaluate_rejects_empty_batches_and_restores_training_mode(as_iterator):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    model.train()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    batches = iter([]) if as_iterator else []

    with pytest.raises(ValueError):
        trainer.evaluate(batches)

    assert model.training is True
