import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import TrainingConfig
from upwash.core.logger import MemoryLogger


@pytest.mark.parametrize(
    ("lr", "expected_weight"),
    [(0.1, 0.4), (0.05, 0.2)],
    ids=["lr-0.1", "lr-0.05"],
)
def test_built_trainer_updates_supplied_model_without_automatic_logging(
    make_optimizer_config, lr, expected_weight
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()
    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=lr)),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )
    inputs = torch.tensor([[1.0]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[2.0]], device="cpu", dtype=torch.float32)

    state = trainer.train([(inputs, targets)])

    # At w=0, mean MSE has gradient -4, giving w=4*lr after one update.
    torch.testing.assert_close(
        model.weight,
        torch.tensor([[expected_weight]], device="cpu", dtype=torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert state.global_step == 1
    assert logger.records == []
