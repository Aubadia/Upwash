"""A real two-process gloo group through setup_runtime / teardown_runtime (task 026).

Each spawned process gets the environment torchrun would set, joins the group,
all-reduces its rank and leaves. The unit tests in tests/unit/core/test_runtime.py
cover the same logic with fakes; this test checks it against real torch.distributed.
"""
import multiprocessing
import os
import socket

import pytest

WORLD_SIZE = 2


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_rank(rank, port, queue):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    try:
        import torch
        import torch.distributed as dist
        from upwash.core.runtime import setup_runtime, teardown_runtime

        runtime = setup_runtime("cpu")
        joined = dist.is_initialized()
        value = torch.tensor([float(runtime.rank + 1)])
        dist.all_reduce(value)  # 1 + 2 on both ranks
        teardown_runtime(runtime)
        queue.put((rank, runtime.rank, runtime.world_size, runtime.device, runtime.is_main,
                   joined, value.item(), dist.is_initialized()))
    except BaseException as error:  # report instead of hanging the parent
        queue.put((rank, repr(error)))
        # Flush the queue's feeder thread before exiting, or the report is lost and
        # the parent only sees its 120 s timeout.
        queue.close()
        queue.join_thread()
        raise


def test_two_cpu_processes_join_communicate_and_leave():
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    port = free_port()
    processes = [context.Process(target=run_rank, args=(r, port, queue)) for r in range(WORLD_SIZE)]
    for p in processes:
        p.start()
    results = {}
    for _ in range(WORLD_SIZE):
        item = queue.get(timeout=120)
        results[item[0]] = item[1:]
    for p in processes:
        p.join(timeout=60)

    for rank in range(WORLD_SIZE):
        assert len(results[rank]) > 1, f"rank {rank} failed: {results[rank]}"
        got_rank, world_size, device, is_main, joined, reduced, still_initialized = results[rank]
        assert (got_rank, world_size, device, is_main) == (rank, WORLD_SIZE, "cpu", rank == 0)
        assert joined
        assert reduced == pytest.approx(3.0)
        assert not still_initialized  # teardown destroyed the group this runtime created
    assert all(p.exitcode == 0 for p in processes)
