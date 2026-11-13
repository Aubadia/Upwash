import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import ScheduleConfig, TrainingConfig
from upwash.core.logger import MemoryLogger


def test_trainer_advances_lr_scheduler_after_each_update_and_logs_used_lr(
    make_optimizer_config, sgd_optimizer_factory,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
    logger = MemoryLogger()

    class HalvingScheduler:
        """Test scheduler isolating Trainer timing from WSD mathematics."""

        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.steps = 0

        def step(self):
            self.steps += 1
            for group in self.optimizer.param_groups:
                group["lr"] *= 0.5

    def lr_scheduler_factory(optimizer, *, config, total_steps):
        return HalvingScheduler(optimizer)

    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1), logging_steps=1),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
        lr_scheduler_factory=lr_scheduler_factory,
        schedule_config=ScheduleConfig(
            warmup_ratio=0.0, decay_ratio=1.0, decay_shape="power", decay_power=1.0,
        ),
        total_steps=2,
        logger=logger,
    )
    batches = [
        (torch.ones(1, 1), torch.full((1, 1), 2.0)),
        (torch.ones(1, 1), torch.full((1, 1), 2.0)),
    ]

    state = trainer.train(batches)

    # w1 = 0 - 0.1*(-4) = 0.4; w2 = 0.4 - 0.05*(-3.2) = 0.56.
    torch.testing.assert_close(model.weight, torch.tensor([[0.56]]), rtol=1e-6, atol=1e-6)
    assert state.global_step == 2
    assert trainer.lr_scheduler.steps == 2
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    assert [record["metrics"]["train/learning_rate"] for record in logger.records] == pytest.approx(
        [0.1, 0.05]
    )
