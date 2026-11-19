"""Runtime: who am I, how many of us, which device; join and leave the process group (task 026).

torchrun sets WORLD_SIZE, RANK and LOCAL_RANK for every process. Process-group calls
are replaced by recording fakes here, so these tests need neither several processes
nor a GPU; tests/integration/test_runtime_process_group.py uses a real gloo group.
The implementation must call them through the `torch.distributed` and `torch.cuda`
modules (e.g. `dist.init_process_group(...)`), which is what the fakes replace.
"""
import pytest
import torch
import torch.distributed as dist

from upwash.core.runtime import Runtime, setup_runtime, teardown_runtime


class CallLog(list):
    """Calls in order; `.state` holds the fake process group's initialized flag."""


@pytest.fixture
def calls(monkeypatch):
    """Record process-group and device calls, in order, instead of performing them."""
    log = CallLog()
    state = {"initialized": False}

    def init_process_group(backend=None, **kwargs):
        log.append(("init", backend))
        state["initialized"] = True

    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(dist, "destroy_process_group", lambda *a, **k: log.append(("destroy",)))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: log.append(("set_device", device)))
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    log.state = state
    return log


def torchrun_env(monkeypatch, *, world_size, rank, local_rank=None):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(rank))
    if local_rank is not None:
        monkeypatch.setenv("LOCAL_RANK", str(local_rank))


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:3"])
def test_without_torchrun_it_is_a_single_process_and_initializes_nothing(calls, device):
    runtime = setup_runtime(device)

    assert runtime == Runtime(rank=0, world_size=1, device=device, initialized_here=False)
    assert runtime.is_main
    assert list(calls) == []


def test_world_size_one_is_also_a_single_process(calls, monkeypatch):
    torchrun_env(monkeypatch, world_size=1, rank=0, local_rank=0)

    assert setup_runtime("cpu") == Runtime(rank=0, world_size=1, device="cpu", initialized_here=False)
    assert list(calls) == []


def test_cpu_ranks_join_a_gloo_group(calls, monkeypatch):
    torchrun_env(monkeypatch, world_size=4, rank=2, local_rank=2)

    runtime = setup_runtime("cpu")

    assert runtime == Runtime(rank=2, world_size=4, device="cpu", initialized_here=True)
    assert not runtime.is_main
    assert list(calls) == [("init", "gloo")]


def test_cuda_ranks_select_their_local_gpu_before_joining_an_nccl_group(calls, monkeypatch):
    # Second node of two: global rank 9, but GPU 1 on this machine.
    torchrun_env(monkeypatch, world_size=16, rank=9, local_rank=1)

    runtime = setup_runtime("cuda")

    assert runtime == Runtime(rank=9, world_size=16, device="cuda:1", initialized_here=True)
    # set_device first, or NCCL may create every rank's communicator on GPU 0.
    assert list(calls) == [("set_device", 1), ("init", "nccl")]


def test_a_configured_cuda_index_is_replaced_by_the_local_rank(calls, monkeypatch):
    torchrun_env(monkeypatch, world_size=8, rank=3, local_rank=3)

    assert setup_runtime("cuda:0").device == "cuda:3"


def test_local_rank_defaults_to_rank(calls, monkeypatch):
    torchrun_env(monkeypatch, world_size=2, rank=1)

    assert setup_runtime("cuda").device == "cuda:1"
    assert ("set_device", 1) in calls


def test_an_existing_process_group_is_joined_not_reinitialized(calls, monkeypatch):
    calls.state["initialized"] = True  # someone else (e.g. a test harness) already initialized it
    torchrun_env(monkeypatch, world_size=2, rank=1, local_rank=1)

    runtime = setup_runtime("cpu")

    assert runtime == Runtime(rank=1, world_size=2, device="cpu", initialized_here=False)
    assert list(calls) == []


def test_teardown_destroys_only_a_group_this_runtime_created(calls):
    teardown_runtime(Runtime())
    teardown_runtime(Runtime(rank=1, world_size=2, device="cpu", initialized_here=False))
    assert list(calls) == []

    teardown_runtime(Runtime(rank=1, world_size=2, device="cpu", initialized_here=True))
    assert list(calls) == [("destroy",)]


def test_runtime_defaults_describe_one_cpu_process():
    runtime = Runtime()

    assert (runtime.rank, runtime.world_size, runtime.device, runtime.initialized_here) == (0, 1, "cpu", False)
    assert runtime.is_main
