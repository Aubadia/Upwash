import torch

from upwash.core.data import DataLoader, TensorDataset
from upwash.core.trainer import Trainer


def test_train_consumes_data_loader_batches():
    # Restore RNG after Linear initialization so other tests are unaffected.
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Two identical samples (x=1, target=2), one per batch.
    dataset = TensorDataset(torch.tensor([[1.0], [1.0]]), torch.tensor([[2.0], [2.0]]))
    loader = DataLoader(dataset, batch_size=1)
    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(reduction="mean"), optimizer=optimizer
    )

    state = trainer.train(loader)

    # Same derivation as the minimal loop: w = 0 -> 0.4 -> 0.72.
    torch.testing.assert_close(
        model.weight, torch.tensor([[0.72]]), rtol=1e-6, atol=1e-6
    )
    assert state.global_step == 2
