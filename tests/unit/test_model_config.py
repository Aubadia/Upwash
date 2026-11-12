import pytest

from upwash.config import ModelConfig


# Reference configurations from the lab spec ("Model Architecture" table):
# (layers, hidden d, MLP m, query heads, KV heads, actual parameters).
# The spec lists 36 KV heads for 150M; only 3 reproduces its parameter count
# and keeps query heads divisible by KV heads (4:1, like the other rows).
SPEC_REFERENCE = {
    "150M": (6, 768, 3_072, 12, 3, 149_825_280),
    "300M": (9, 1_024, 5_184, 16, 4, 298_273_792),
    "500M": (11, 1_536, 4_672, 24, 6, 498_731_520),
    "750M": (13, 1_792, 5_888, 28, 7, 745_749_760),
    "1B": (15, 1_792, 8_000, 28, 7, 995_432_704),
}

MODEL_150M = {
    "vocab_size": 128_256,
    "n_layers": 6,
    "d_model": 768,
    "d_ff": 3_072,
    "n_heads": 12,
    "n_kv_heads": 3,
    "max_seq_len": 4096,
    "rope_theta": 500_000.0,
    "norm_eps": 1e-5,
    "embedding_std": 0.1,
    "output_std_coeff": 1.0,
}


def make_model_config(**overrides):
    return ModelConfig(**{**MODEL_150M, **overrides})


def spec_config(scale):
    n_layers, d_model, d_ff, n_heads, n_kv_heads, _ = SPEC_REFERENCE[scale]
    return make_model_config(n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_kv_heads=n_kv_heads)


def test_model_config_exposes_architecture_and_initialization():
    config = make_model_config()

    assert config.vocab_size == 128_256
    assert config.n_layers == 6
    assert config.d_model == 768
    assert config.d_ff == 3_072
    assert config.n_heads == 12
    assert config.n_kv_heads == 3
    assert config.max_seq_len == 4096
    assert config.rope_theta == 500_000.0
    assert config.norm_eps == 1e-5
    assert config.embedding_std == 0.1
    assert config.output_std_coeff == 1.0


@pytest.mark.parametrize("missing", list(MODEL_150M))
def test_model_config_requires_every_setting(missing):
    fields = {name: value for name, value in MODEL_150M.items() if name != missing}

    with pytest.raises(TypeError, match=rf"missing .*'{missing}'"):
        ModelConfig(**fields)


@pytest.mark.parametrize("scale", list(SPEC_REFERENCE))
def test_model_config_num_parameters_matches_the_spec_reference_table(scale):
    # Tied embeddings counted once, bias-free projections, two RMSNorms per
    # layer plus a final RMSNorm, vocabulary 128,256 (the spec's assumptions).
    assert spec_config(scale).num_parameters() == SPEC_REFERENCE[scale][-1]


@pytest.mark.parametrize(("scale", "head_dim"), [("150M", 64), ("500M", 64), ("1B", 64)])
def test_model_config_head_dim_is_d_model_over_query_heads(scale, head_dim):
    assert spec_config(scale).head_dim == head_dim


@pytest.mark.parametrize("n_kv_heads", [12, 1], ids=["multi-head", "multi-query"])
def test_model_config_accepts_any_kv_head_count_that_divides_query_heads(n_kv_heads):
    assert make_model_config(n_kv_heads=n_kv_heads).n_kv_heads == n_kv_heads


@pytest.mark.parametrize(
    "name", ["vocab_size", "n_layers", "d_model", "d_ff", "n_heads", "n_kv_heads", "max_seq_len"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_model_config_rejects_non_positive_sizes(name, value):
    # Checked before the divisibility rules, so n_heads=0 is a ValueError, not a ZeroDivisionError.
    with pytest.raises(ValueError):
        make_model_config(**{name: value})


def test_model_config_rejects_d_model_not_divisible_by_query_heads():
    with pytest.raises(ValueError):
        make_model_config(d_model=770)  # 770 / 12 is not an integer head dimension


@pytest.mark.parametrize("n_kv_heads", [5, 36], ids=["five", "spec-table-typo"])
def test_model_config_rejects_query_heads_not_divisible_by_kv_heads(n_kv_heads):
    # Each KV head must serve a whole number of query heads.
    with pytest.raises(ValueError):
        make_model_config(n_kv_heads=n_kv_heads)


def test_model_config_rejects_an_odd_head_dim():
    # 780 / 12 = 65: standard RoPE rotates dimension pairs, so head_dim must be even.
    with pytest.raises(ValueError):
        make_model_config(d_model=780)


@pytest.mark.parametrize("name", ["rope_theta", "norm_eps", "embedding_std", "output_std_coeff"])
@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "infinity"],
)
def test_model_config_rejects_non_positive_or_nonfinite_scalars(name, value):
    with pytest.raises(ValueError):
        make_model_config(**{name: value})


def test_model_config_to_dict_returns_every_setting():
    # Recorded in the run config: the spec requires model configuration and initialization.
    assert make_model_config().to_dict() == MODEL_150M
