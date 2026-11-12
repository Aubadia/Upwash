import torch

from upwash.builder import build_trainer
from upwash.config import TrainingConfig
from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer


def test_build_trainer_constructs_configured_optimizer_and_preserves_supplied_objects(make_optimizer_config):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    initial_weight = model.weight.detach().clone()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    logger = MemoryLogger()
    config = TrainingConfig(optimizer=make_optimizer_config(lr=0.1))

    trainer = build_trainer(
        config=config,
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=torch.optim.SGD,
        logger=logger,
    )

    assert isinstance(trainer, Trainer)
    assert trainer.model is model
    assert trainer.loss_fn is loss_fn
    assert trainer.logger is logger
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    # The optimizer factory receives the learning rate only; Muon/AdamW-specific
    # settings are for the Muon + AdamW builder (task 018), not for SGD.
    assert [group["lr"] for group in trainer.optimizer.param_groups] == [0.1]
    torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)
    assert logger.records == []


def test_build_trainer_allows_omitting_logger(make_optimizer_config):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)

    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1)),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=torch.optim.SGD,
    )

    assert trainer.logger is None
