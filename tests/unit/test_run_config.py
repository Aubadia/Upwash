"""JSON run configs: round trip and validation."""

import json

import pytest

from upwash.config import (
    DataConfig, LoggingConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig, load_run_config,
)


def make_config(make_optimizer_config, **overrides):
    settings = dict(
        model=ModelConfig(
            vocab_size=128256, n_layers=6, d_model=768, d_ff=3072, n_heads=12, n_kv_heads=3,
            max_seq_len=4096, rope_theta=500_000.0, norm_eps=1e-5,
            embedding_std=0.1, output_std_coeff=1.0,
        ),
        train_data=DataConfig("/data/train", "*.bin", "bin", 4096, 4096 * 64, shuffle_seed=0),
        eval_data=DataConfig("/data/dev", "*.bin", "bin", 4096, 4096 * 8),
        training=TrainingConfig(
            optimizer=make_optimizer_config(), logging_steps=10, eval_steps=100,
            precision="bf16", max_grad_norm=1.0,
        ),
        schedule=ScheduleConfig(warmup_ratio=0.001, decay_ratio=0.1, decay_shape="power", decay_power=1.0),
        batch_size=16, eval_batch_size=8, device="cuda", seed=3,
        logging=LoggingConfig(project="lab6-predictable-scaling", name="run", group="sweep-lr"),
        output_dir="/out/run", tokenizer="meta-llama/Llama-3.2-1B", tokenizer_revision="abc",
        export_max_position_embeddings=8192,
    )
    settings.update(overrides)
    return RunConfig(**settings)


def test_run_config_round_trips_through_json(tmp_path, make_optimizer_config):
    config = make_config(make_optimizer_config)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()))

    loaded = load_run_config(path)

    assert loaded.to_dict() == config.to_dict()
    assert isinstance(loaded.training, TrainingConfig)
    assert loaded.training.max_grad_norm == 1.0
    assert loaded.train_data.shuffle_seed == 0
    assert loaded.logging.group == "sweep-lr"


def test_optional_sections_may_be_null(tmp_path, make_optimizer_config):
    config = make_config(make_optimizer_config, eval_data=None, logging=None, output_dir=None)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()))

    loaded = load_run_config(path)

    assert loaded.eval_data is None and loaded.logging is None and loaded.output_dir is None


def test_unknown_key_is_rejected(tmp_path, make_optimizer_config):
    data = make_config(make_optimizer_config).to_dict()
    data["learning_rate"] = 1e-3
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data))

    with pytest.raises(TypeError, match="learning_rate"):
        load_run_config(path)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_max_grad_norm_must_be_positive_and_finite(make_optimizer_config, value):
    with pytest.raises(ValueError, match="max_grad_norm"):
        TrainingConfig(optimizer=make_optimizer_config(), max_grad_norm=value)
