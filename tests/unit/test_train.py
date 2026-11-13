"""Entry-point contracts without constructing a real HF Llama."""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def make_run_config(tmp_path, make_optimizer_config, **overrides):
    from upwash.config import DataConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig

    np.save(tmp_path / "tokens.npy", np.arange(21, dtype=np.int32) % 8)
    settings = dict(
        model=ModelConfig(
            vocab_size=8, n_layers=1, d_model=4, d_ff=8,
            n_heads=2, n_kv_heads=1, max_seq_len=4,
            rope_theta=500_000.0, norm_eps=1e-5,
            embedding_std=0.1, output_std_coeff=1.0,
        ),
        train_data=DataConfig(
            root=str(tmp_path), pattern="tokens.npy", format="npy",
            seq_len=4, num_tokens=20,
        ),
        eval_data=None,
        training=TrainingConfig(optimizer=make_optimizer_config(), precision="fp32"),
        schedule=ScheduleConfig(
            warmup_ratio=0.0, decay_ratio=0.0, decay_shape="power", decay_power=1.0,
        ),
        batch_size=2, eval_batch_size=2, device="cpu", seed=17, logging=None,
    )
    settings.update(overrides)
    return RunConfig(**settings)


def test_run_config_composes_existing_configs(tmp_path, make_optimizer_config):
    config = make_run_config(tmp_path, make_optimizer_config)

    assert config.model.vocab_size == 8
    assert config.train_data.num_tokens == 20
    assert config.training.optimizer.lr == 0.002
    assert config.schedule.decay_ratio == 0.0
    assert config.eval_data is None
    assert config.logging is None


def test_run_config_rejects_a_partial_training_sequence(tmp_path, make_optimizer_config):
    from upwash.config import DataConfig

    data = DataConfig(
        root=str(tmp_path), pattern="tokens.npy", format="npy",
        seq_len=4, num_tokens=21,
    )
    with pytest.raises(ValueError, match="num_tokens.*seq_len"):
        make_run_config(tmp_path, make_optimizer_config, train_data=data)


def test_train_runs_the_entire_budget_including_the_final_partial_batch(
    tmp_path, make_optimizer_config, monkeypatch,
):
    config = make_run_config(tmp_path, make_optimizer_config)
    entry = importlib.import_module("upwash.train")
    initial = {}

    class TinyLM(torch.nn.Module):
        """Cheap HF-output double; real gradients and optimizer updates."""

        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.hidden = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

        def forward(self, inputs):
            return SimpleNamespace(logits=self.lm_head(self.hidden(self.embed_tokens(inputs))))

    def build_model(model_config):
        assert model_config is config.model
        model = TinyLM()
        initial.update({name: p.detach().clone() for name, p in model.named_parameters()})
        return model

    monkeypatch.setattr(entry, "build_llama_model", build_model)
    # Preserve the test runner's RNG; train is responsible for applying config.seed.
    with torch.random.fork_rng(devices=[]):
        result = entry.train(config)

    assert isinstance(result, entry.TrainResult)
    assert result.state.global_step == 3  # five sequences -> batches [2, 2, 1]
    assert isinstance(result.model, TinyLM)
    assert result.model.lm_head.weight is result.model.embed_tokens.weight
    for name, parameter in result.model.named_parameters():
        assert parameter.device.type == "cpu"
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter.detach(), initial[name]), name


@pytest.fixture
def recording_sdk(monkeypatch):
    from upwash.core.logger import flock

    run = SimpleNamespace(events=[], finished=0, settings=[])

    def log(data, *, step, commit):
        run.events.append((dict(data), step, commit))

    def finish():
        run.finished += 1

    def init(**settings):
        run.settings.append(settings)
        return run

    run.log = log
    run.finish = finish
    monkeypatch.setattr(flock.wandb, "init", init)
    monkeypatch.setattr(flock.atexit, "register", lambda callback: None)
    return run


def setup_logged_run(tmp_path, make_optimizer_config, monkeypatch):
    from upwash.config import DataConfig, LoggingConfig

    config = make_run_config(
        tmp_path, make_optimizer_config,
        logging=LoggingConfig(project="test-project", name="test-run"),
    )
    config.training.logging_steps = 2
    config.training.eval_steps = 2
    np.save(tmp_path / "dev.npy", np.arange(9, dtype=np.int32) % 8)
    config.eval_data = DataConfig(str(tmp_path), "dev.npy", "npy", 4, 8)
    entry = importlib.import_module("upwash.train")

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 8)

        def forward(self, inputs):
            return SimpleNamespace(logits=self.embed_tokens(inputs))

    monkeypatch.setattr(entry, "build_llama_model", lambda config: TinyLM())
    return entry, config


def test_train_logs_metrics_and_finishes_its_flock_run(
    tmp_path, make_optimizer_config, monkeypatch, recording_sdk,
):
    entry, config = setup_logged_run(tmp_path, make_optimizer_config, monkeypatch)
    with torch.random.fork_rng(devices=[]):
        result = entry.train(config)

    assert result.state.global_step == 3
    assert recording_sdk.settings == [dict(project="test-project", name="test-run", config={})]
    assert recording_sdk.finished == 1
    assert [step for _, step, _ in recording_sdk.events] == [2, 3]
    assert [metrics["train/total_tokens_seen"] for metrics, _, _ in recording_sdk.events] == [16, 20]
    for metrics, _, commit in recording_sdk.events:
        assert commit is True
        assert metrics["eval/perplexity"] == pytest.approx(math.exp(metrics["eval/loss"]))
        for key in ("train/loss", "train/grad_norm", "train/learning_rate", "train/tokens_per_second"):
            assert math.isfinite(metrics[key])


@pytest.mark.parametrize("failure_at", ["builder", "training"])
def test_train_finishes_its_flock_run_when_execution_fails(
    tmp_path, make_optimizer_config, monkeypatch, recording_sdk, failure_at,
):
    entry, config = setup_logged_run(tmp_path, make_optimizer_config, monkeypatch)
    error = RuntimeError("execution failed")

    def fail(*args, **kwargs):
        raise error

    if failure_at == "builder":
        monkeypatch.setattr(entry, "build_trainer", fail)
    else:
        monkeypatch.setattr(entry, "causal_lm_loss", fail)
    with torch.random.fork_rng(devices=[]):
        with pytest.raises(RuntimeError) as observed:
            entry.train(config)
    assert observed.value is error
    assert len(recording_sdk.settings) == 1
    assert recording_sdk.finished == 1


def test_train_with_logging_disabled_does_not_initialize_sdk(
    tmp_path, make_optimizer_config, monkeypatch, recording_sdk,
):
    entry, config = setup_logged_run(tmp_path, make_optimizer_config, monkeypatch)
    config.logging = None
    with torch.random.fork_rng(devices=[]):
        entry.train(config)
    assert recording_sdk.settings == []
    assert recording_sdk.events == []
    assert recording_sdk.finished == 0
