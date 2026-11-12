import pytest


# Starting point of the 150M sweep: the first value of every grid in the
# experiment plan (roadmap/017-optimizer-schedule-config.md).
SWEEP_START_OPTIMIZER = {
    "lr": 0.002,
    "muon_beta": 0.9,
    "adam_beta1": 0.85,
    "adam_beta2": 0.95,
    "adam_lr_multiplier": 1.5,
    "weight_decay": 0.04,
}


@pytest.fixture
def sweep_start_optimizer():
    return dict(SWEEP_START_OPTIMIZER)


@pytest.fixture
def make_optimizer_config():
    """Builds a valid OptimizerConfig; every field is required, so tests override only what they test."""

    from upwash.config import OptimizerConfig

    def make(**overrides):
        return OptimizerConfig(**{**SWEEP_START_OPTIMIZER, **overrides})

    return make
