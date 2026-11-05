import torch

from upwash.core.trainer import Trainer


def test_compute_loss_uses_supplied_mse_loss():
    model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)
    trainer = Trainer(model=model, loss_fn=torch.nn.MSELoss(reduction="mean"))

    loss = trainer.compute_loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    torch.testing.assert_close(
        loss,
        torch.tensor(4.0, device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )
