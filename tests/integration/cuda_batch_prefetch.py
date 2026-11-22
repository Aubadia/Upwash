"""Run directly with ./test.sh tests/integration/cuda_batch_prefetch.py."""

import numpy as np
import pytest
import torch

from upwash.config import DataConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig
from upwash.core.trainer import Trainer
from upwash.train import train


@pytest.mark.parametrize(
    ("prefetch_options", "batch_device"),
    [({}, "cpu"), ({"prefetch_to_device": True}, "cuda:0")],
    ids=["default-loader", "cuda-prefetch"],
)
def test_train_receives_ordered_batches_on_the_expected_device(
    tmp_path, make_optimizer_config, monkeypatch, prefetch_options, batch_device,
):
    assert torch.cuda.is_available(), "This integration test requires a CUDA device"
    tokens = torch.arange(21) % 32
    np.save(tmp_path / "tokens.npy", tokens.numpy().astype(np.int32))
    expected_inputs = tokens[:-1].reshape(-1, 4).split(2)
    expected_targets = tokens[1:].reshape(-1, 4).split(2)
    config = RunConfig(
        model=ModelConfig(
            vocab_size=32, n_layers=1, d_model=16, d_ff=32, n_heads=2, n_kv_heads=1,
            max_seq_len=4, rope_theta=500_000.0, norm_eps=1e-5,
            embedding_std=0.1, output_std_coeff=1.0,
        ),
        train_data=DataConfig(str(tmp_path), "tokens.npy", "npy", 4, 20),
        training=TrainingConfig(optimizer=make_optimizer_config()),
        schedule=ScheduleConfig(
            warmup_ratio=0.0, decay_ratio=0.0, decay_shape="power", decay_power=1.0,
        ),
        batch_size=2, device="cuda:0", seed=17, **prefetch_options,
    )
    original_train = Trainer.train

    def observe_training_batches(self, batches, *, eval_batches=None):
        assert len(batches) == 3

        def checked_batches():
            for (inputs, targets), want_inputs, want_targets in zip(
                batches, expected_inputs, expected_targets, strict=True,
            ):
                assert inputs.device == targets.device == torch.device(batch_device)
                torch.testing.assert_close(inputs.cpu(), want_inputs)
                torch.testing.assert_close(targets.cpu(), want_targets)
                yield inputs, targets

        return original_train(self, checked_batches(), eval_batches=eval_batches)

    monkeypatch.setattr(Trainer, "train", observe_training_batches)
    with torch.random.fork_rng(devices=[0]):
        result = train(config)

    assert result.state.global_step == 3
