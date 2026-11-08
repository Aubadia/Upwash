import pytest

from upwash.config import OptimizerConfig, TrainingConfig


def test_training_config_exposes_configured_optimizer_learning_rate():
    config = TrainingConfig(optimizer=OptimizerConfig(lr=0.1))

    assert config.optimizer.lr == 0.1


def test_optimizer_config_rejects_negative_learning_rate():
    with pytest.raises(ValueError):
        OptimizerConfig(lr=-0.1)


def test_optimizer_config_accepts_zero_learning_rate():
    config = OptimizerConfig(lr=0.0)

    assert config.lr == 0.0


@pytest.mark.parametrize(
    "lr",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_optimizer_config_rejects_nonfinite_learning_rate(lr):
    with pytest.raises(ValueError):
        OptimizerConfig(lr=lr)
