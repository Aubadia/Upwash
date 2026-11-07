import pytest
import torch

from upwash.core.trainer import Trainer


@pytest.mark.parametrize("batch_factory", [list, lambda: iter([])], ids=["list", "iterator"])
def test_train_rejects_empty_batches_without_updating_parameters(batch_factory):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    initial_weight = model.weight.detach().clone()
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    with pytest.raises(ValueError):
        trainer.train(batch_factory())

    torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)
