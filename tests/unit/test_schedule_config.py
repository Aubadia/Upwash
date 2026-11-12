import pytest

from upwash.config import ScheduleConfig


# Starting point of the 150M sweep: warmup 0.001, decay 0.1, linear decay to zero.
SWEEP_START_SCHEDULE = {
    "warmup_ratio": 0.001,
    "decay_ratio": 0.1,
    "decay_shape": "power",
    "decay_power": 1.0,
}


def make_schedule(**overrides):
    return ScheduleConfig(**{**SWEEP_START_SCHEDULE, **overrides})


def test_schedule_config_exposes_warmup_and_decay_settings():
    config = ScheduleConfig(warmup_ratio=0.01, decay_ratio=0.3, decay_shape="cosine", decay_power=1.0)

    assert config.warmup_ratio == 0.01
    assert config.decay_ratio == 0.3
    assert config.decay_shape == "cosine"
    assert config.decay_power == 1.0


@pytest.mark.parametrize("missing", list(SWEEP_START_SCHEDULE))
def test_schedule_config_requires_every_setting(missing):
    fields = {name: value for name, value in SWEEP_START_SCHEDULE.items() if name != missing}

    # The error must name the missing setting, not reject some other argument.
    with pytest.raises(TypeError, match=rf"missing .*'{missing}'"):
        ScheduleConfig(**fields)


@pytest.mark.parametrize(
    ("warmup_ratio", "decay_ratio"),
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.01, 0.5)],
    ids=["no-warmup-no-decay", "all-warmup", "all-decay", "sum-exactly-one", "largest-grid-values"],
)
def test_schedule_config_accepts_ratios_that_fit_in_the_run(warmup_ratio, decay_ratio):
    config = make_schedule(warmup_ratio=warmup_ratio, decay_ratio=decay_ratio)

    assert (config.warmup_ratio, config.decay_ratio) == (warmup_ratio, decay_ratio)


@pytest.mark.parametrize("name", ["warmup_ratio", "decay_ratio"])
@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan")],
    ids=["negative", "above-one", "nan"],
)
def test_schedule_config_rejects_a_ratio_outside_zero_to_one(name, value):
    with pytest.raises(ValueError):
        make_schedule(**{name: value})


def test_schedule_config_rejects_warmup_and_decay_longer_than_the_run():
    # Warmup and decay are fractions of the same total steps; together they can't exceed it.
    with pytest.raises(ValueError):
        make_schedule(warmup_ratio=0.6, decay_ratio=0.5)


@pytest.mark.parametrize("shape", ["cosine", "power"])
def test_schedule_config_accepts_cosine_and_power_decay(shape):
    assert make_schedule(decay_shape=shape).decay_shape == shape


@pytest.mark.parametrize("shape", ["linear", "Cosine", "step", ""])
def test_schedule_config_rejects_other_decay_shapes(shape):
    # Linear decay is spelled power with decay_power=1.0, so each schedule has one spelling.
    with pytest.raises(ValueError):
        make_schedule(decay_shape=shape)


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test_schedule_config_accepts_the_swept_decay_powers(power):
    assert make_schedule(decay_power=power).decay_power == power


@pytest.mark.parametrize(
    "power",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "infinity"],
)
def test_schedule_config_rejects_a_non_positive_or_nonfinite_decay_power(power):
    with pytest.raises(ValueError):
        make_schedule(decay_power=power)


def test_schedule_config_to_dict_returns_every_setting():
    assert make_schedule().to_dict() == SWEEP_START_SCHEDULE
