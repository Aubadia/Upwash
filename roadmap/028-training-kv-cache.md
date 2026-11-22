# 028 — Disable the training KV cache

Branch `feat/acceleration`, based on `main` at `e2efe61`. The behavior comes
from the `experiment` Llama adapter at `180bd4f`.

## Behavior

A Llama built with `build_llama_model(config)` returns `past_key_values=None`
on a training forward without an explicit `use_cache` argument. Upwash trains
on independent packed batches and does not reuse the cache between forwards.
The existing builder signature and Trainer call remain sufficient.

## First Red — 2026-11-20

The test runs a real two-layer Llama forward in training mode on CPU with a
single four-token input. The test passed with the `experiment` adapter loaded
from `180bd4f` in a separate process, using Transformers 5.22.2 and PyTorch
2.15.0+cu132. The same versions were used for the local regression.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_llama_init.py tests/unit/test_adaptor.py tests/unit/test_train.py
```

Exit 1, **1 failed, 27 passed in 4.67s**. The forward succeeds, but returns
`DynamicCache(layers=[DynamicLayer, DynamicLayer])` where the test expects
`None`. The adapter currently inherits the Hugging Face cache default.

Tests ran directly on the user's machine. The CPU Red has no execution blocker.
In this session NVML reports GPU access blocked by the operating system and
PyTorch reports CUDA unavailable. GPU acceptance was not run. The
[FlashAttention-3 build used by the experiment](https://github.com/Aubadia/Upwash/blob/8718eb1/scripts/build_fa3.sbatch)
disables SM8x kernels. The upstream source can include those kernels when
built with different flags. See [031](031-flashattention-3.md) for laptop
preflight results.

Next, the user disables KV caching in the Llama training config, then rerun
the targeted regression to verify Green.

## Verification after adapter wiring — 2026-11-20

The user added a required `ModelConfig.use_cache` field, and the adapter
forwards it into `LlamaConfig`. The same CPU regression command exited 1 with
**26 failed, 2 passed in 8.40s**. All 26 failures occurred at existing
`ModelConfig` calls that omit `use_cache`. The cache regression fails before
the training forward with `TypeError: ModelConfig.__init__() missing 1 required
positional argument: 'use_cache'`.

A separate CPU forward with explicit `use_cache=False` returned `None`, and
one with explicit `use_cache=True` returned `DynamicCache`. The adapter honors
the explicit setting. Next, preserve existing callers and disable training
caching by default, then rerun the regression. If keeping the config field,
also verify the `ModelConfig.to_dict()` contract because it uses `asdict`.

## Green — 2026-11-20

The user gave `ModelConfig.use_cache` a default of `False`, restoring existing
construction. The KV cache regression passes. The original serialization test
was the sole failure in the expanded regression (84 passed) because its expected
dictionary lacked `use_cache`. The test still checks exact dictionary equality
and now covers both the default and explicit opt-in.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_llama_init.py tests/unit/test_adaptor.py tests/unit/test_train.py tests/unit/test_model_config.py
```

Exit 0, **85 passed in 5.36s**. Running the same command without the four file
arguments ran the full suite and returned **394 passed, 3 failed in 5.56s**.
The three failures could not open a socket in the sandbox (`PermissionError`)
and passed when rerun with local loopback socket access:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/integration/test_ddp_training.py tests/integration/test_runtime_process_group.py
```

Exit 0, **3 passed in 19.11s**. No GPU tests were run. Next proposal: agree on
the config interface for opt-in model-only `torch.compile` before adding its
entry-point test.
