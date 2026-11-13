import pytest

from upwash.config import OptimizerConfig, TrainingConfig

OPTIMIZER_FIELDS = ["lr", "muon_beta", "adam_beta1", "adam_beta2", "adam_lr_multiplier", "weight_decay"]


def test_training_config_exposes_configured_optimizer_learning_rate(make_optimizer_config):
    config = TrainingConfig(optimizer=make_optimizer_config(lr=0.1))

    assert config.optimizer.lr == 0.1


def test_optimizer_config_rejects_negative_learning_rate(make_optimizer_config):
    with pytest.raises(ValueError):
        make_optimizer_config(lr=-0.1)


def test_optimizer_config_accepts_zero_learning_rate(make_optimizer_config):
    config = make_optimizer_config(lr=0.0)

    assert config.lr == 0.0


@pytest.mark.parametrize(
    "lr",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_optimizer_config_rejects_nonfinite_learning_rate(make_optimizer_config, lr):
    with pytest.raises(ValueError):
        make_optimizer_config(lr=lr)


# --- Task 017: Muon + AdamW settings -------------------------------------------------


def test_optimizer_config_exposes_muon_and_adamw_settings():
    config = OptimizerConfig(
        lr=0.008,
        muon_beta=0.95,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_lr_multiplier=1.6,
        weight_decay=0.16,
    )

    assert config.lr == 0.008
    assert config.muon_beta == 0.95
    assert config.adam_beta1 == 0.9
    assert config.adam_beta2 == 0.98
    assert config.adam_lr_multiplier == 1.6
    # Stored as the coefficient c; the builder applies c / d_model later.
    assert config.weight_decay == 0.16


@pytest.mark.parametrize("missing", OPTIMIZER_FIELDS)
def test_optimizer_config_requires_every_setting(sweep_start_optimizer, missing):
    fields = {name: value for name, value in sweep_start_optimizer.items() if name != missing}

    # No implicit defaults: a sweep config that forgets a value fails loudly.
    # The error must name the missing setting, not reject some other argument.
    with pytest.raises(TypeError, match=rf"missing .*'{missing}'"):
        OptimizerConfig(**fields)


@pytest.mark.parametrize("name", ["muon_beta", "adam_beta1", "adam_beta2"])
@pytest.mark.parametrize("value", [0.0, 0.5, 0.99])
def test_optimizer_config_accepts_betas_in_zero_to_one(make_optimizer_config, name, value):
    config = make_optimizer_config(**{name: value})

    assert getattr(config, name) == value


@pytest.mark.parametrize("name", ["muon_beta", "adam_beta1", "adam_beta2"])
@pytest.mark.parametrize(
    "value",
    [-0.1, 1.0, 1.5, float("nan")],
    ids=["negative", "one", "above-one", "nan"],
)
def test_optimizer_config_rejects_betas_outside_zero_to_one(make_optimizer_config, name, value):
    # A beta of 1 would keep the momentum forever and never use new gradients.
    with pytest.raises(ValueError):
        make_optimizer_config(**{name: value})


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "infinity"],
)
def test_optimizer_config_rejects_a_non_positive_or_nonfinite_adam_lr_multiplier(make_optimizer_config, value):
    with pytest.raises(ValueError):
        make_optimizer_config(adam_lr_multiplier=value)


def test_optimizer_config_accepts_zero_weight_decay(make_optimizer_config):
    assert make_optimizer_config(weight_decay=0.0).weight_decay == 0.0


@pytest.mark.parametrize(
    "value",
    [-0.04, float("nan"), float("inf")],
    ids=["negative", "nan", "infinity"],
)
def test_optimizer_config_rejects_negative_or_nonfinite_weight_decay(make_optimizer_config, value):
    with pytest.raises(ValueError):
        make_optimizer_config(weight_decay=value)


def test_optimizer_config_to_dict_returns_every_setting(make_optimizer_config, sweep_start_optimizer):
    # Used to record the run's settings, e.g. FlockLogger(config=...).
    assert make_optimizer_config().to_dict() == sweep_start_optimizer
