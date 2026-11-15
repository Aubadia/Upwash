"""Two CPU ranks (gloo) through the real entry point match one process on the global batch."""

import multiprocessing
import os
import socket

import numpy as np
import pytest
import torch

from upwash.config import DataConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig

WORLD_SIZE = 2


def make_config(root, *, batch_size, make_optimizer_config):
    return RunConfig(
        model=ModelConfig(
            vocab_size=32, n_layers=1, d_model=16, d_ff=32, n_heads=2, n_kv_heads=1,
            max_seq_len=8, rope_theta=500_000.0, norm_eps=1e-5, embedding_std=0.1, output_std_coeff=1.0,
        ),
        # 16 samples of 8 tokens, shuffled: four updates of a global batch of 4.
        train_data=DataConfig(str(root), "tokens.npy", "npy", 8, 8 * 16, shuffle_seed=5),
        eval_data=None,
        training=TrainingConfig(
            optimizer=make_optimizer_config(lr=0.04), precision="fp32",
            max_grad_norm=1.0,
        ),
        schedule=ScheduleConfig(warmup_ratio=0.25, decay_ratio=0.5, decay_shape="power", decay_power=1.0),
        batch_size=batch_size, eval_batch_size=1, device="cpu", seed=11, logging=None,
    )


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_rank(rank, port, config, queue):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    from upwash.train import train

    try:
        result = train(config)
        # numpy copies: shared-memory tensors do not survive the child's exit.
        queue.put((rank, {n: p.detach().numpy().copy() for n, p in result.model.named_parameters()},
                   result.state.global_step))
    except BaseException as error:  # report instead of hanging the parent
        queue.put((rank, repr(error), None))
        raise


@pytest.fixture
def token_root(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "tokens.npy", rng.integers(0, 32, size=8 * 16 + 1, dtype=np.int32))
    return tmp_path


def test_two_ranks_match_one_process_on_the_global_batch(token_root, make_optimizer_config, monkeypatch):
    from upwash.train import train

    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    reference = train(make_config(token_root, batch_size=4, make_optimizer_config=make_optimizer_config))
    expected = {n: p.detach() for n, p in reference.model.named_parameters()}

    # Per-rank batch 2 on each of two ranks: the same global batch of 4.
    config = make_config(token_root, batch_size=2, make_optimizer_config=make_optimizer_config)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    port = free_port()
    processes = [context.Process(target=run_rank, args=(r, port, config, queue)) for r in range(WORLD_SIZE)]
    for process in processes:
        process.start()
    results = dict()
    for _ in range(WORLD_SIZE):
        rank, parameters, steps = queue.get(timeout=300)
        results[rank] = (parameters, steps)
    for process in processes:
        process.join(timeout=60)

    from upwash.adaptor import build_llama_model

    torch.manual_seed(config.seed)
    initial = {n: p.detach() for n, p in build_llama_model(config.model).named_parameters()}

    for rank, (parameters, steps) in results.items():
        assert isinstance(parameters, dict), f"rank {rank} failed: {parameters}"
        assert steps == reference.state.global_step == 4
        assert parameters.keys() == expected.keys()
    for name in expected:
        # DDP keeps replicas identical.
        np.testing.assert_array_equal(results[0][0][name], results[1][0][name], err_msg=name)
        # torch.optim.Muon orthogonalizes in bfloat16 (torch/optim/_muon.py), so
        # fp32-level differences between all-reduced and full-batch gradients can
        # move hidden matrices by bf16-quantized steps (~1e-4 observed); AdamW
        # parameters agree to ~1e-6.
        observed = torch.from_numpy(results[0][0][name])
        torch.testing.assert_close(observed, expected[name], rtol=0, atol=5e-4, msg=name)
        assert (expected[name] - initial[name]).abs().max() > 10 * 5e-4, f"{name} barely trained"
    assert all(process.exitcode == 0 for process in processes)
