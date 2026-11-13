import torch

from upwash.builder import build_trainer
from upwash.config import ScheduleConfig, TrainingConfig
from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer


def test_build_trainer_constructs_and_injects_configured_lr_scheduler(
    make_optimizer_config, sgd_optimizer_factory,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
    initial_weight = model.weight.detach().clone()
    schedule_config = ScheduleConfig(
        warmup_ratio=0.1,
        decay_ratio=0.3,
        decay_shape="power",
        decay_power=2.0,
    )

    class SchedulerStub:
        def step(self):
            raise AssertionError("Builder must not advance the scheduler")

    scheduler = SchedulerStub()
    supplied_optimizers = []

    def lr_scheduler_factory(optimizer, *, config, total_steps):
        assert config is schedule_config
        assert total_steps == 100
        supplied_optimizers.append(optimizer)
        return scheduler

    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1)),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
        lr_scheduler_factory=lr_scheduler_factory,
        schedule_config=schedule_config,
        total_steps=100,
    )

    assert supplied_optimizers == [trainer.optimizer]
    assert trainer.lr_scheduler is scheduler
    torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)


def test_build_trainer_passes_model_and_complete_config_to_muon_factory(make_optimizer_config):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 2, bias=False, device="cpu", dtype=torch.float32)
    initial_weight = model.weight.detach().clone()
    optimizer_config = make_optimizer_config(
        lr=0.008,
        muon_beta=0.95,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_lr_multiplier=1.6,
        weight_decay=0.16,
    )
    created_optimizers = []

    def muon_factory(supplied_model, *, config):
        assert supplied_model is model
        assert config is optimizer_config
        optimizer = torch.optim.Muon(
            supplied_model.parameters(),
            lr=config.lr,
            momentum=config.muon_beta,
            weight_decay=config.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        created_optimizers.append(optimizer)
        return optimizer

    trainer = build_trainer(
        config=TrainingConfig(optimizer=optimizer_config),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=muon_factory,
    )

    assert trainer.optimizer is created_optimizers[0]
    assert isinstance(trainer.optimizer, torch.optim.Muon)
    group = trainer.optimizer.param_groups[0]
    assert group["params"][0] is model.weight
    assert group["lr"] == 0.008
    assert group["momentum"] == 0.95
    assert group["weight_decay"] == 0.16
    assert group["adjust_lr_fn"] == "match_rms_adamw"
    torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)


def test_build_trainer_constructs_configured_optimizer_and_preserves_supplied_objects(make_optimizer_config, sgd_optimizer_factory):
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
        optimizer_factory=sgd_optimizer_factory,
        logger=logger,
    )

    assert isinstance(trainer, Trainer)
    assert trainer.model is model
    assert trainer.loss_fn is loss_fn
    assert trainer.logger is logger
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    # The test adapter selects SGD settings from the complete optimizer config.
    assert [group["lr"] for group in trainer.optimizer.param_groups] == [0.1]
    torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)
    assert logger.records == []


def test_build_trainer_allows_omitting_logger(make_optimizer_config, sgd_optimizer_factory):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)

    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1)),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
    )

    assert trainer.logger is None


# --- Task 020: precision --------------------------------------------------------------


def test_build_trainer_passes_the_configured_precision_to_the_trainer(
    make_optimizer_config, sgd_optimizer_factory,
):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, bias=False)
    dtypes = []
    model.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))

    trainer = build_trainer(
        config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1), precision="bf16"),
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer_factory=sgd_optimizer_factory,
    )
    trainer.train([(torch.ones(4, 2), torch.ones(4, 1))])

    assert dtypes == [torch.bfloat16]
