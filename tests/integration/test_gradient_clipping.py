import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import TrainingConfig
from upwash.core.logger import MemoryLogger


@pytest.mark.parametrize("use_logger", [True, False], ids=["with-logger", "without-logger"])
def test_built_trainer_clips_global_gradient_and_logs_original_norm(
    make_optimizer_config, sgd_optimizer_factory, use_logger,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    logger = MemoryLogger() if use_logger else None
    trainer = build_trainer(
        config=TrainingConfig(
            optimizer=make_optimizer_config(lr=0.1),
            logging_steps=1,
            max_grad_norm=1.0,
        ),
        model=model,
        loss_fn=torch.nn.MSELoss(),
        optimizer_factory=sgd_optimizer_factory,
        logger=logger,
    )

    inputs = torch.tensor([[0.75]], dtype=torch.float32, device="cpu")
    targets = torch.tensor([[2.0]], dtype=torch.float32, device="cpu")
    trainer.train([(inputs, targets)])

    # MSE gives weight and bias gradients of -3 and -4. Clip their joint norm
    # of 5 to 1, then use the clipped gradients for the SGD update.
    assert model.weight.item() == pytest.approx(0.06)
    assert model.bias.item() == pytest.approx(0.08)
    if logger is not None:
        assert logger.records[0]["metrics"]["train/grad_norm"] == pytest.approx(5.0)


def test_disabled_clipping_preserves_finite_gradients_with_an_overflowing_norm(
    make_optimizer_config, sgd_optimizer_factory,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 2, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()

    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1)),
        model=model,
        loss_fn=lambda predictions, targets: (predictions * targets).sum(),
        optimizer_factory=sgd_optimizer_factory,
    )
    inputs = torch.tensor([[1.0]], dtype=torch.float32, device="cpu")
    targets = torch.full((1, 2), 1e30, dtype=torch.float32, device="cpu")

    trainer.train([(inputs, targets)])

    # Both raw gradients are finite even though their FP32 L2 norm overflows.
    torch.testing.assert_close(model.weight, -0.1 * targets.T)
