import torch

from upwash.core.data import DataLoader, TensorDataset
from upwash.core.trainer import Trainer


def test_repeated_training_fits_a_small_linear_regression():
    # End-to-end scenario: real data loader, model, loss, and optimizer together.
    # Noise-free data from y = 3x + 1, so the exact solution is w=3, b=1.
    inputs = torch.linspace(-1.0, 1.0, 16, device="cpu", dtype=torch.float32).reshape(16, 1)
    targets = 3 * inputs + 1
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    # Restore RNG after initialization; zero parameters make the run deterministic.
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    def full_data_loss():
        with torch.no_grad():
            return trainer.compute_loss(inputs, targets).item()

    initial_loss = full_data_loss()
    # Each train() call is one pass (epoch) over 16 samples in batches of 4.
    for _ in range(30):
        state = trainer.train(loader)
        assert state.global_step == 4
    final_loss = full_data_loss()

    # A standalone PyTorch run of this setup reaches loss ~2e-9 and w ~ 2.99993,
    # so these thresholds leave a wide margin while still requiring real learning.
    assert initial_loss > 4.0
    assert final_loss < 1e-4
    torch.testing.assert_close(model.weight, torch.tensor([[3.0]]), rtol=0, atol=1e-3)
    torch.testing.assert_close(model.bias, torch.tensor([1.0]), rtol=0, atol=1e-3)
