"""train() wiring for data parallelism, without spawning processes (task 026, step 4).

A fake Runtime makes this process look like one rank of two. DistributedDataParallel,
the barrier and teardown are replaced by recording fakes, and a cheap TinyLM stands in
for the HF Llama, so the tests check the wiring only:

- the training data is sharded by rank,
- the model is wrapped once with the agreed DDP settings,
- only rank 0 creates the Flock run,
- every rank reaches the barrier and the teardown, also on failure,
- the unwrapped model is returned.

tests/integration/test_ddp_training.py runs two real gloo ranks.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from upwash.core.runtime import Runtime

SEQ_LEN, NUM_SAMPLES = 4, 8  # 8 sequences: 2 global batches of 2 ranks x 2


class TinyLM(torch.nn.Module):
    """Cheap HF-output double with real gradients; records every input batch it sees."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(64, 64)  # logits over the vocabulary
        self.seen = []

    def forward(self, inputs):
        if self.training:
            self.seen.append(inputs.detach().clone())
        return SimpleNamespace(logits=self.embed_tokens(inputs))


class FakeDDP(torch.nn.Module):
    instances = []

    def __init__(self, module, **kwargs):
        super().__init__()
        self.module, self.kwargs = module, kwargs
        FakeDDP.instances.append(self)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def make_config(tmp_path, make_optimizer_config, *, logging=None):
    from upwash.config import DataConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig

    # Distinct token ids, so each sequence identifies which sample it is.
    np.save(tmp_path / "tokens.npy", np.arange(NUM_SAMPLES * SEQ_LEN + 1, dtype=np.int32))
    return RunConfig(
        model=ModelConfig(vocab_size=64, n_layers=1, d_model=4, d_ff=8, n_heads=2, n_kv_heads=1,
                          max_seq_len=SEQ_LEN, rope_theta=500_000.0, norm_eps=1e-5,
                          embedding_std=0.1, output_std_coeff=1.0),
        train_data=DataConfig(str(tmp_path), "tokens.npy", "npy", SEQ_LEN, NUM_SAMPLES * SEQ_LEN),
        eval_data=None,
        training=TrainingConfig(optimizer=make_optimizer_config(), precision="fp32"),
        schedule=ScheduleConfig(warmup_ratio=0.0, decay_ratio=0.0, decay_shape="power", decay_power=1.0),
        batch_size=2, eval_batch_size=2, device="cpu", seed=17, logging=logging,
    )


@pytest.fixture
def entry(monkeypatch):
    """upwash.train with a TinyLM and recording fakes for DDP, barrier and teardown."""
    module = importlib.import_module("upwash.train")
    calls = []
    models = []

    def build_model(config):
        models.append(TinyLM())
        return models[-1]

    FakeDDP.instances = []
    monkeypatch.setattr(module, "build_llama_model", build_model)
    # Whichever import style train.py uses, these names resolve to the fakes.
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    monkeypatch.setattr(module, "DistributedDataParallel", FakeDDP, raising=False)
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: calls.append("barrier"))
    monkeypatch.setattr(module, "barrier", lambda *a, **k: calls.append("barrier"), raising=False)

    def teardown(runtime):
        calls.append(("teardown", runtime))

    import upwash.core.runtime as runtime_module
    monkeypatch.setattr(runtime_module, "teardown_runtime", teardown)
    monkeypatch.setattr(module, "teardown_runtime", teardown, raising=False)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(module=module, calls=calls, models=models)


def as_rank(entry, monkeypatch, rank, world_size=2):
    runtime = Runtime(rank=rank, world_size=world_size, device="cpu", initialized_here=False)
    import upwash.core.runtime as runtime_module
    monkeypatch.setattr(runtime_module, "setup_runtime", lambda device: runtime)
    monkeypatch.setattr(entry.module, "setup_runtime", lambda device: runtime, raising=False)
    return runtime


@pytest.fixture
def flock_runs(monkeypatch):
    from upwash.core.logger import flock

    runs = []

    def init(**settings):
        runs.append(settings)
        return SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)

    monkeypatch.setattr(flock.wandb, "init", init)
    monkeypatch.setattr(flock.atexit, "register", lambda callback: None)
    return runs


def sample(dataset, i):
    return dataset[i][0]


def test_a_single_process_trains_unwrapped_and_unsharded(tmp_path, make_optimizer_config, entry):
    config = make_config(tmp_path, make_optimizer_config)

    with torch.random.fork_rng(devices=[]):
        result = entry.module.train(config)

    assert FakeDDP.instances == []
    assert "barrier" not in entry.calls
    assert result.state.global_step == NUM_SAMPLES // 2  # all 8 samples, batch 2
    assert result.model is entry.models[0]
    assert sum(len(batch) for batch in result.model.seen) == NUM_SAMPLES


def test_rank_one_of_two_trains_its_shard_through_ddp(tmp_path, make_optimizer_config, entry, monkeypatch):
    from upwash.builder import build_dataset

    config = make_config(tmp_path, make_optimizer_config)
    runtime = as_rank(entry, monkeypatch, rank=1)

    with torch.random.fork_rng(devices=[]):
        result = entry.module.train(config)

    # Global batches [0..3] and [4..7]; rank 1 takes samples 2, 3 and 6, 7.
    data = build_dataset(config.train_data)
    expected = [torch.stack([sample(data, 2), sample(data, 3)]),
                torch.stack([sample(data, 6), sample(data, 7)])]
    assert result.state.global_step == 2  # per-rank updates; also the WSD horizon
    assert len(result.model.seen) == 2
    for got, want in zip(result.model.seen, expected, strict=True):
        torch.testing.assert_close(got, want)

    assert len(FakeDDP.instances) == 1
    wrapper = FakeDDP.instances[0]
    assert wrapper.module is entry.models[0]
    assert wrapper.kwargs.get("device_ids") is None  # CPU: no device ids
    assert wrapper.kwargs.get("forward_sync_buffers") is False
    assert result.model is entry.models[0]  # unwrapped: no "module." parameter prefix
    assert "barrier" in entry.calls
    assert entry.calls[-1] == ("teardown", runtime)


@pytest.mark.parametrize(("rank", "expected_runs"), [(0, 1), (1, 0)])
def test_only_rank_zero_creates_the_flock_run(tmp_path, make_optimizer_config, entry, monkeypatch,
                                              flock_runs, rank, expected_runs):
    from upwash.config import LoggingConfig

    config = make_config(tmp_path, make_optimizer_config,
                         logging=LoggingConfig(project="test-project", name="test-run"))
    as_rank(entry, monkeypatch, rank=rank)

    with torch.random.fork_rng(devices=[]):
        entry.module.train(config)

    assert len(flock_runs) == expected_runs


def test_every_rank_tears_down_when_training_fails(tmp_path, make_optimizer_config, entry, monkeypatch):
    config = make_config(tmp_path, make_optimizer_config)
    runtime = as_rank(entry, monkeypatch, rank=1)
    error = RuntimeError("training failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(entry.module, "build_trainer", fail)
    with torch.random.fork_rng(devices=[]):
        with pytest.raises(RuntimeError) as observed:
            entry.module.train(config)

    assert observed.value is error
    assert entry.calls[-1] == ("teardown", runtime)


def test_a_dataset_that_does_not_fill_global_batches_is_rejected_on_several_ranks(
        tmp_path, make_optimizer_config, entry, monkeypatch):
    from upwash.config import DataConfig

    config = make_config(tmp_path, make_optimizer_config)
    # 6 sequences do not fill global batches of 4; one process would accept them.
    config.train_data = DataConfig(str(tmp_path), "tokens.npy", "npy", SEQ_LEN, 6 * SEQ_LEN)
    as_rank(entry, monkeypatch, rank=0)

    with torch.random.fork_rng(devices=[]):
        with pytest.raises(ValueError, match="global batch"):
            entry.module.train(config)
