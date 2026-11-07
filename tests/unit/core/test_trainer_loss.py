import torch

from upwash.core.trainer import Trainer


def test_compute_loss_uses_supplied_mse_loss():
    model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)
    # Supply the required optimizer; compute_loss itself performs no update.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(reduction="mean"), optimizer=optimizer
    )

    loss = trainer.compute_loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    torch.testing.assert_close(
        loss,
        torch.tensor(4.0, device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_loss_uses_supplied_l1_loss():
    model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)
    # Supply the required optimizer; compute_loss itself performs no update.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model, loss_fn=torch.nn.L1Loss(reduction="mean"), optimizer=optimizer
    )

    loss = trainer.compute_loss(inputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    torch.testing.assert_close(
        loss,
        torch.tensor(2.0, device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_loss_uses_output_modified_by_model_forward_hook():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model, loss_fn=torch.nn.MSELoss(reduction="mean"), optimizer=optimizer
    )
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)

    # A public module hook changes prediction 0 to 1, so MSE becomes (1-2)^2=1.
    handle = model.register_forward_hook(lambda module, args, output: output + 1.0)
    try:
        loss = trainer.compute_loss(inputs, targets)
    finally:
        handle.remove()

    torch.testing.assert_close(
        loss,
        torch.tensor(1.0, device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )
