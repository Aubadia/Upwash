"""Two real gloo ranks through train() match one process on the global batch (task 026, step 4).

Each rank trains a tiny real Llama with Muon + AdamW and WSD on a per-rank batch of 2,
the single process on a batch of 4. The comparison runs with and without gradient clipping.

Tolerance, measured 2026-11-19 with a scratch reference composition of main's components
(RankShard + DDP + build_trainer), no clipping:
- replicas are bit-identical;
- the largest difference from one process is 1.3e-4, in a Muon matrix. torch.optim.Muon
  orthogonalizes in bfloat16, so fp32-level differences between all-reduced and full-batch
  gradients can move a matrix by a bf16-quantized step. AdamW parameters differ by ~1e-6;
- the smallest training movement is 3.4e-2.
atol 5e-4 is ~4x the largest observed difference and ~70x below the movement.
"""
import multiprocessing
import os
import socket

import numpy as np
import pytest
import torch

WORLD_SIZE = 2
ATOL = 5e-4


def make_config(root, *, batch_size, make_optimizer_config, max_grad_norm):
    from upwash.config import DataConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig

    return RunConfig(
        model=ModelConfig(vocab_size=32, n_layers=1, d_model=16, d_ff=32, n_heads=2, n_kv_heads=1,
                          max_seq_len=8, rope_theta=500_000.0, norm_eps=1e-5, embedding_std=0.1,
                          output_std_coeff=1.0),
        # 16 samples of 8 tokens: four updates of a global batch of 4.
        train_data=DataConfig(str(root), "tokens.npy", "npy", 8, 8 * 16),
        eval_data=None,
        training=TrainingConfig(
            optimizer=make_optimizer_config(lr=0.04), precision="fp32", max_grad_norm=max_grad_norm,
        ),
        schedule=ScheduleConfig(warmup_ratio=0.25, decay_ratio=0.5, decay_shape="power", decay_power=1.0),
        batch_size=batch_size, eval_batch_size=1, device="cpu", seed=11, logging=None,
    )


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_rank(rank, port, config, queue):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    try:
        torch.set_num_threads(1)
        import torch.distributed as dist
        from upwash.train import train

        result = train(config)
        # numpy copies: shared-memory tensors do not survive the child's exit.
        queue.put((rank, {n: p.detach().numpy().copy() for n, p in result.model.named_parameters()},
                   result.state.global_step, dist.is_initialized()))
    except BaseException as error:  # report instead of hanging the parent
        queue.put((rank, repr(error), None, None))
        queue.close()
        queue.join_thread()
        raise


@pytest.fixture
def token_root(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "tokens.npy", rng.integers(0, 32, size=8 * 16 + 1, dtype=np.int32))
    return tmp_path


@pytest.mark.parametrize("max_grad_norm", [None, 0.1], ids=["no-clipping", "clipping"])
def test_two_ranks_match_one_process_on_the_global_batch(
    token_root, make_optimizer_config, monkeypatch, max_grad_norm,
):
    from upwash.adaptor import build_llama_model
    from upwash.train import train

    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        reference = train(make_config(
            token_root, batch_size=4, make_optimizer_config=make_optimizer_config, max_grad_norm=max_grad_norm,
        ))
    if max_grad_norm is not None:
        final_grad_norm = torch.stack([p.grad.norm() for p in reference.model.parameters() if p.grad is not None]).norm()
        assert final_grad_norm.item() == pytest.approx(max_grad_norm, rel=2e-5)
    expected = {n: p.detach() for n, p in reference.model.named_parameters()}

    # Per-rank batch 2 on each of two ranks: the same global batch of 4.
    config = make_config(
        token_root, batch_size=2, make_optimizer_config=make_optimizer_config, max_grad_norm=max_grad_norm,
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    port = free_port()
    processes = [context.Process(target=run_rank, args=(r, port, config, queue)) for r in range(WORLD_SIZE)]
    for process in processes:
        process.start()
    results = {}
    for _ in range(WORLD_SIZE):
        rank, parameters, steps, still_initialized = queue.get(timeout=300)
        results[rank] = (parameters, steps, still_initialized)
    for process in processes:
        process.join(timeout=60)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        initial = {n: p.detach() for n, p in build_llama_model(config.model).named_parameters()}

    for rank, (parameters, steps, still_initialized) in results.items():
        assert isinstance(parameters, dict), f"rank {rank} failed: {parameters}"
        assert steps == reference.state.global_step == 4
        assert parameters.keys() == expected.keys()  # unwrapped: no "module." prefix
        assert not still_initialized  # train() tore down the group it created
    for name in expected:
        # DDP keeps replicas identical.
        np.testing.assert_array_equal(results[0][0][name], results[1][0][name], err_msg=name)
        torch.testing.assert_close(torch.from_numpy(results[0][0][name]), expected[name],
                                   rtol=0, atol=ATOL, msg=name)
        assert (expected[name] - initial[name]).abs().max() > 10 * ATOL, f"{name} barely trained"
    assert all(process.exitcode == 0 for process in processes)
