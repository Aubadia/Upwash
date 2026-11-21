# 032 — CUDA batch prefetch

This task follows FlashAttention-3 backend selection. The experiment at
`8718eb1` measured `2,045,180 tokens/s` with prefetch versus `1,973,828
tokens/s` in the matched warm 150M, eight-GPU baseline. See
[the experiment results](https://github.com/Aubadia/Upwash/blob/8718eb1/experiments/speed150m/RESULTS.md).

## Agreed interface

`RunConfig.prefetch_to_device` defaults to `False`. When enabled on CUDA,
`train()` wraps the training loader after rank sharding and batching. The
wrapper prepares one batch ahead using pinned host memory and a separate
CUDA copy stream. It yields batches on the runtime device in their original
order and preserves length, partial final batches, and repeatable traversal.
CPU runs continue to use the ordinary loader. The first increment covers the
training loader.

## CUDA reference smoke — 2026-11-20

PyTorch `2.15.0+cu132` detected the local RTX 3080 Laptop GPU and supported
pinned-host transfers on a separate CUDA stream. A standalone probe loaded
the experiment's `prefetched` and `DeviceBatches` definitions from `8718eb1`.
The GPU consumer verified the exact tensor values and order during two
traversals of four batches, including a partial final batch. The reported
length and empty input also passed. The Python stdin command used
`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -`
with local GPU access and exited 0. Laptop throughput remains unmeasured.

## First Red — 2026-11-20

The explicitly invoked CUDA integration observes the loader at `Trainer.train`
and then delegates to real training. It checks that prefetch yields ordered
CUDA tensors, with loader length three and batch sizes 2, 2, and 1. The default
case checks the same contents on CPU before the Trainer's normal device move.
Both cases are expected to complete three real Llama optimizer updates. The
CUDA file follows the existing explicit invocation convention for GPU tests.

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/integration/cuda_batch_prefetch.py
```

Exit 1 on the RTX 3080 Laptop, **1 failed, 1 passed in 7.03s**. Default CUDA
training and ordered CPU loader batches pass. The opt-in case stops at
`TypeError: RunConfig.__init__() got an unexpected keyword argument 'prefetch_to_device'`.
The CUDA batch delivery assertions have not executed for the opt-in path.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/core/test_trainer_device.py tests/integration/test_training_with_data_loader.py
```

Exit 0, **25 passed in 9.23s**. The CUDA integration required the config field
and loader wrapper before it could reach the opt-in assertions.

## Green — 2026-11-20

`prefetch_to_device=True` wraps the CUDA training loader after rank sharding
and batching. A background worker pins host tensors and submits the next
batch on a dedicated copy stream. Each batch has its own CUDA event. The
consumer waits for that event and registers the tensors on its stream so
their storage remains safe while in use. The default, CPU, and evaluation
paths continue using the ordinary loader.

Reran the CUDA integration command above on the RTX 3080 Laptop with local
GPU access. Exit 0, **2 passed in 6.75s**. Both paths completed three real
Llama updates while preserving batch order and the partial tail. A separate
CUDA probe confirmed pinned nonblocking transfers on a separate stream,
submission of the next batch before requesting it, successful traversal
after closing an iterator early, and empty input. A CPU smoke with the flag
enabled completed three updates through the ordinary loader. The standalone
probes ran via `.venv/bin/python` with `PYTHONPATH=src`, `OMP_NUM_THREADS=2`,
`PYTHONDONTWRITEBYTECODE=1`, and device visibility set for each case.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short
```

Exit 0 with local loopback access, **409 passed in 41.62s**. Next, measure
actual GPU copy/compute overlap and end-to-end throughput against the default
loader. Multi-GPU prefetch has not been verified.
