"""Entry-point contracts without constructing a real HF Llama."""

import importlib
import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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


def test_run_config_defaults_to_eager_with_only_one_compile_switch(tmp_path, make_optimizer_config):
    config = make_run_config(tmp_path, make_optimizer_config)

    assert config.compile_scope == "none"
    with pytest.raises(TypeError, match="compile_model"):
        make_run_config(tmp_path, make_optimizer_config, compile_model=True)


@pytest.mark.parametrize("use_autocast", [True, False], ids=["autocast", "bf16-projection"])
def test_train_selects_dense_fa3_wrapper_without_cast_warning(
    tmp_path, make_optimizer_config, monkeypatch, use_autocast,
):
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers import modeling_flash_attention_utils as flash_utils
    from transformers.integrations.flash_attention import flash_attention_forward

    from upwash.core.state import TrainingState

    backend = "upwash_fa3"
    config = make_run_config(tmp_path, make_optimizer_config, attention_implementation=backend)
    q = torch.arange(32, dtype=torch.float32).reshape(2, 2, 4, 2)
    k = torch.arange(16, dtype=torch.float32).reshape(2, 1, 4, 2)
    v = (k + 16).to(torch.bfloat16)
    expected = q.transpose(1, 2).to(torch.bfloat16)
    state = TrainingState()
    dense_call = Mock()

    def native_dense(query, key, value, *, softmax_scale=None, causal=False):
        for actual, original in zip((query, key, value), (q, k, v), strict=True):
            torch.testing.assert_close(actual, original.transpose(1, 2).to(torch.bfloat16))
        assert causal and softmax_scale == 0.5
        dense_call()
        return query

    native = ModuleType("flash_attn_interface")
    native.flash_attn_func = native_dense
    native.flash_attn_varlen_func = Mock(side_effect=AssertionError("Dense batches use dense FA3"))
    native.flash_attn_with_kvcache = None
    monkeypatch.setitem(sys.modules, native.__name__, native)
    warning = Mock()

    for registry in (AttentionInterface, AttentionMaskInterface):
        monkeypatch.setattr(registry, "_global_mapping", registry._global_mapping.copy())
    monkeypatch.setattr(flash_utils.logger, "warning_once", warning)
    with monkeypatch.context() as reference:
        for cache_name in (
            "_loaded_implementation", "_flash_fn", "_flash_varlen_fn", "_flash_with_kvcache_fn",
            "_pad_fn", "_unpad_fn", "_process_flash_kwargs_fn", "_process_with_kvcache_kwargs_fn",
        ):
            reference.setattr(flash_utils, cache_name, None)
        with torch.random.fork_rng(devices=[]):
            reference_layer = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        reference_layer.config = SimpleNamespace(_attn_implementation="flash_attention_3")
        reference_layer.is_causal = True
        reference_output, reference_weights = flash_attention_forward(
            reference_layer, q.clone(), k.clone(), v.clone(), None, scaling=0.5,
        )
        torch.testing.assert_close(reference_output, expected)
        assert reference_weights is None
        warning.assert_called_once()
        dense_call.assert_called_once()
    warning.reset_mock()
    dense_call.reset_mock()

    def build_trainer(*, model, **kwargs):
        def run(*args, **kwargs):
            layer = model.model.layers[0].self_attn
            if not use_autocast:
                layer.to(dtype=torch.bfloat16)
            attention = AttentionInterface()[model.config._attn_implementation]
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_autocast):
                output, weights = attention(layer, q.clone(), k.clone(), v.clone(), None, scaling=0.5)
            torch.testing.assert_close(output, reference_output)
            assert weights is None
            warning.assert_not_called()
            return state

        return SimpleNamespace(train=run)

    entry = importlib.import_module("upwash.train")
    monkeypatch.setattr(entry, "build_trainer", build_trainer)
    with torch.random.fork_rng(devices=[]):
        result = entry.train(config)

    assert result.state is state
    dense_call.assert_called_once()


@pytest.mark.parametrize(
    ("compile_options", "expected_mode"),
    [
        ({}, None),
        ({"compile_scope": "none", "compile_mode": "reduce-overhead"}, None),
        ({"compile_scope": "model"}, "default"),
        ({"compile_scope": "model", "compile_mode": "reduce-overhead"}, "reduce-overhead"),
    ],
    ids=["eager-by-default", "eager-with-mode", "default-mode", "explicit-mode"],
)
def test_train_uses_model_compilation_only_when_enabled(
    tmp_path, make_optimizer_config, monkeypatch, compile_options, expected_mode,
):
    from upwash.core.state import TrainingState

    entry = importlib.import_module("upwash.train")
    config = make_run_config(tmp_path, make_optimizer_config, **compile_options)
    model = torch.nn.Identity()
    compile_calls = []
    state = TrainingState()
    inputs = torch.tensor([1.0])
    expected_output = inputs if expected_mode is None else inputs + 1

    def compile_forward(module, *, mode=None, **kwargs):
        compile_calls.append(mode)
        return lambda inputs: module(inputs) + 1

    def build_trainer(*, model, **kwargs):
        def train(*args, **kwargs):
            torch.testing.assert_close(model(inputs), expected_output)
            return state

        return SimpleNamespace(train=train)

    monkeypatch.setattr(entry, "build_llama_model", lambda config: model)
    monkeypatch.setattr(torch, "compile", compile_forward)
    monkeypatch.setattr(entry, "build_trainer", build_trainer)

    with torch.random.fork_rng(devices=[]):
        result = entry.train(config)

    assert compile_calls == ([] if expected_mode is None else [expected_mode])
    assert result.model is model
    assert result.state is state


def test_train_compiles_model_and_loss_together_in_objective_scope(
    tmp_path, make_optimizer_config, monkeypatch,
):
    entry = importlib.import_module("upwash.train")
    config = make_run_config(
        tmp_path, make_optimizer_config,
        batch_size=5, compile_scope="objective",
    )
    in_compiled_call = False
    calls = []
    compile_modes = []
    original_loss = torch.nn.functional.cross_entropy

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 8)

        def forward(self, inputs):
            calls.append(("model", in_compiled_call))
            return SimpleNamespace(logits=self.embed_tokens(inputs))

    def loss(*args, **kwargs):
        calls.append(("loss", in_compiled_call))
        return original_loss(*args, **kwargs)

    def record_compiled_call(function):
        def compiled(*args, **kwargs):
            nonlocal in_compiled_call
            in_compiled_call = True
            try:
                return function(*args, **kwargs)
            finally:
                in_compiled_call = False

        return compiled

    def compile_objective(function, *, mode=None, **kwargs):
        compile_modes.append(mode)
        if isinstance(function, torch.nn.Module):
            # Preserve the module interface when the compiler receives a module.
            monkeypatch.setattr(function, "forward", record_compiled_call(function.forward))
            return function
        return record_compiled_call(function)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TinyLM()
        initial_weight = model.embed_tokens.weight.detach().clone()
        monkeypatch.setattr(entry, "build_llama_model", lambda config: model)
        monkeypatch.setattr(torch.nn.functional, "cross_entropy", loss)
        monkeypatch.setattr(torch, "compile", compile_objective)
        result = entry.train(config)

    assert compile_modes == ["default"]
    assert calls == [("model", True), ("loss", True)]
    assert result.state.global_step == 1
    assert result.model is model
    assert not torch.equal(model.embed_tokens.weight, initial_weight)


@pytest.mark.parametrize(
    ("compile_scope", "expected_settings"),
    [
        ("none", (8, False)),
        ("model", (64, True)),
        ("objective", (64, True)),
    ],
)
def test_train_scopes_dynamo_recompile_settings_to_compiled_training(
    tmp_path, make_optimizer_config, monkeypatch, compile_scope, expected_settings,
):
    from torch._dynamo import config as dynamo_config

    from upwash.core.state import TrainingState

    entry = importlib.import_module("upwash.train")
    config = make_run_config(tmp_path, make_optimizer_config, compile_scope=compile_scope)
    state = TrainingState()

    def run(train_loader, *, eval_batches):
        settings = (dynamo_config.recompile_limit, dynamo_config.fail_on_recompile_limit_hit)
        assert settings == expected_settings
        return state

    monkeypatch.setattr(entry, "build_llama_model", lambda config: torch.nn.Identity())
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kwargs: None)
    monkeypatch.setattr(entry, "build_trainer", lambda **kwargs: SimpleNamespace(train=run))

    with torch.random.fork_rng(devices=[]), dynamo_config.patch(
        recompile_limit=8, fail_on_recompile_limit_hit=False,
    ):
        result = entry.train(config)
        assert (dynamo_config.recompile_limit, dynamo_config.fail_on_recompile_limit_hit) == (8, False)

    assert result.state is state


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


def test_public_train_uses_stream_and_resumes_global_progress(tmp_path, make_optimizer_config):
    from contextlib import contextmanager

    from upwash.adaptor import build_llama_model
    from upwash.core.trainer import TrainingProgress
    from upwash.train import RunHooks, train

    config = make_run_config(tmp_path, make_optimizer_config, train_data=None)
    config.training.logging_steps = 1
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    events = []
    logger = Mock()
    built_model = build_llama_model(config.model)

    @contextmanager
    def batches(runtime):
        events.append(("open", runtime.rank))
        try:
            yield [(inputs, targets), (inputs, targets)]
        finally:
            events.append(("close", runtime.rank))

    def build_optimizer(model, *, config):
        return torch.optim.SGD(model.parameters(), lr=config.lr)

    def prepare(runtime, model, trainer):
        assert runtime.rank == 0
        assert model is built_model
        assert trainer.lr_scheduler.total_steps == 5
        trainer.lr_scheduler.load_state_dict({"update": 4})
        trainer.logger = logger
        return TrainingProgress(step=3, tokens=12)

    def after_step(progress):
        events.append(progress)
        return {"custom_metric": float(progress.tokens)}

    hooks = RunHooks(
        batches=batches, build_model=lambda _: built_model, build_optimizer=build_optimizer,
        prepare=prepare, after_step=after_step, schedule_steps=5,
    )
    result = train(config, hooks=hooks)

    assert result.model is built_model
    assert result.state.global_step == 5
    assert events == [
        ("open", 0), TrainingProgress(step=4, tokens=16),
        TrainingProgress(step=5, tokens=20), ("close", 0),
    ]
    assert [call.kwargs["step"] for call in logger.log.call_args_list] == [4, 5]
    assert [call.args[0]["custom_metric"] for call in logger.log.call_args_list] == [16.0, 20.0]
    assert [call.args[0]["train/total_tokens_seen"] for call in logger.log.call_args_list] == [16, 20]
