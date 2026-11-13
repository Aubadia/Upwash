import pytest

from upwash.config import DataConfig


# The 150M run on Flock's pre-tokenized shards: 200 tokens per parameter.
FLOCK_150M = {
    "root": "/data/fineweb-edu-llama3",
    "pattern": "*/tokens.npy",
    "format": "npy",
    "seq_len": 4096,
    "num_tokens": 30_000_000_000,
}


def make_data_config(**overrides):
    return DataConfig(**{**FLOCK_150M, **overrides})


def test_data_config_exposes_source_and_token_budget():
    config = DataConfig(
        root="/data/tokens",
        pattern="sample/10BT/*.bin",
        format="bin",
        seq_len=2048,
        num_tokens=60_000_000_000,
    )

    assert config.root == "/data/tokens"
    assert config.pattern == "sample/10BT/*.bin"
    assert config.format == "bin"
    assert config.seq_len == 2048
    assert config.num_tokens == 60_000_000_000


@pytest.mark.parametrize("missing", list(FLOCK_150M))
def test_data_config_requires_every_setting(missing):
    fields = {name: value for name, value in FLOCK_150M.items() if name != missing}

    with pytest.raises(TypeError, match=rf"missing .*'{missing}'"):
        DataConfig(**fields)


def test_data_config_does_not_touch_the_file_system():
    # Settings only: the path is checked when the dataset is built, not here.
    config = make_data_config(root="/does/not/exist")

    assert config.root == "/does/not/exist"


@pytest.mark.parametrize("fmt", ["npy", "bin"])
def test_data_config_accepts_npy_and_bin_formats(fmt):
    assert make_data_config(format=fmt).format == fmt


@pytest.mark.parametrize("fmt", ["NPY", "parquet", "u4", ""])
def test_data_config_rejects_other_formats(fmt):
    with pytest.raises(ValueError):
        make_data_config(format=fmt)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_data_config_rejects_a_non_positive_seq_len(seq_len):
    with pytest.raises(ValueError):
        make_data_config(seq_len=seq_len, num_tokens=4096)


def test_data_config_accepts_a_budget_of_exactly_one_sample():
    assert make_data_config(seq_len=4096, num_tokens=4096).num_tokens == 4096


@pytest.mark.parametrize("num_tokens", [4095, 0, -1])
def test_data_config_rejects_a_budget_smaller_than_one_sample(num_tokens):
    # Fewer than seq_len tokens would train on zero samples.
    with pytest.raises(ValueError):
        make_data_config(seq_len=4096, num_tokens=num_tokens)


def test_data_config_to_dict_returns_every_setting():
    assert make_data_config().to_dict() == FLOCK_150M
