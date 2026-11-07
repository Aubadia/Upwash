import pytest
import torch

from upwash.core.trainer import Trainer


class TrainingModeLinear(torch.nn.Linear):
    """Record the mode during real forward computation."""

    def __init__(self):
        super().__init__(1, 1, bias=False, device="cpu", dtype=torch.float32)
        self.forward_training_modes = []

    def forward(self, inputs):
        # Observe mode when predictions are computed, not just after train returns.
        self.forward_training_modes.append(self.training)
        return super().forward(inputs)


@pytest.mark.parametrize(
    ("batch_count", "expected_weight"),
    [(1, 0.4), (2, 0.72)],
    ids=["one-sgd-update", "two-updates-with-fresh-gradients"],
)
def test_train_updates_parameters_in_training_mode_and_returns_progress(
    batch_count, expected_weight
):
    # Linear initialization consumes randomness even though we overwrite its weight.
    # Restore the RNG afterward so this test does not affect other tests.
    with torch.random.fork_rng(devices=[]):
        model = TrainingModeLinear()
    with torch.no_grad():
        model.weight.zero_()
    # Start in eval mode so Trainer must actively enable training mode.
    model.eval()
    # Real SGD exercises backward and parameter updates without optimizer extras.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0
    )
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)
    # A single-use iterator checks support for iterable batches without indexing.
    batches = iter(
        [(inputs.clone(), targets.clone()) for _ in range(batch_count)]
    )
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=optimizer,
    )

    state = trainer.train(batches)

    # With x=1 and target=2, loss=(w-2)^2 and d(loss)/dw=2*(w-2).
    # SGD: first w=0-0.1*(-4)=0.4, then w=0.4-0.1*(-3.2)=0.72.
    # If gradients accumulate, the second weight becomes 1.12 instead of 0.72.
    torch.testing.assert_close(
        model.weight,
        torch.tensor([[expected_weight]], device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    # Progress counts completed optimizer updates for this train call.
    assert state.global_step == batch_count
    # Require a forward observation: all([]) alone would pass without any forward.
    assert model.forward_training_modes
    assert all(model.forward_training_modes)


def test_train_counts_steps_per_call_and_continues_updating_parameters():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=optimizer,
    )
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)

    # Inspect each result immediately; returned-state snapshot semantics are deferred.
    for expected_weight in (0.4, 0.72):
        state = trainer.train(iter([(inputs.clone(), targets.clone())]))

        assert state.global_step == 1
        # Reset progress only: the second call continues from the trained weight.
        torch.testing.assert_close(
            model.weight,
            torch.tensor([[expected_weight]], device="cpu", dtype=torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )
