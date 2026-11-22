# 031 — FlashAttention-3 selection

The user chose FA3 before CUDA batch prefetch. The behavioral reference is
the experiment at `8718eb1`. It sets the HF attention backend after placing
the model on the runtime device. Its FA3 build pins `flash-attn-3` to upstream
commit `f63c472c4dc19237c0635d6a629df065ff64796e` and sets
`FLASH_ATTENTION_DISABLE_SM80=TRUE`.

## Agreed interface

`RunConfig.attention_implementation: str | None = None`. `None` preserves
the model's Transformers default. An explicit backend, including
`"flash_attention_3"`, is sent to `model.set_attn_implementation()` after
device placement and before wrapping or compilation. The first test checks
this entry-point behavior with CPU test doubles. CUDA numerical acceptance
requires the FA3 extension.

## Laptop preflight — 2026-11-20

The laptop's RTX 3080 is SM 8.6. The experiment binary excludes SM8x, but the
[pinned upstream setup](https://raw.githubusercontent.com/Dao-AILab/flash-attention/f63c472c4dc19237c0635d6a629df065ff64796e/hopper/setup.py)
can include SM80 forward and backward variants. The installed Transformers
5.22.2 compatibility matrix allows FA3 on CUDA compute-capability major 8
and above. The upstream README highlights Hopper. Actual laptop FA3
execution remains unverified.

The Python stdin preflight used
`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
with local GPU access. A real HF Llama raises `ImportError` when selecting
`flash_attention_3` because the extension is absent. The same model runs SDPA
forward and backward in CUDA BF16 with finite gradients. PyTorch's extension
preflight reports a CUDA major-version mismatch between system `nvcc` 12.8
and the PyTorch `2.15.0+cu132` build. The diagnostic exits 0 after checking
those two expected environment errors. No dependencies were installed.

## First Red — 2026-11-20

The CPU wiring test simulates a two-rank run and records model placement,
backend selection, compilation, and DDP wrapping. The default case expects
the normal Transformers backend selection. The explicit case expects
`flash_attention_3` after placement and before compilation and wrapping.
The setter, compiler, and DDP are test doubles. The tiny model still performs
two real CPU optimizer updates when the path reaches Trainer.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train_ddp.py tests/unit/test_train.py tests/unit/test_adaptor.py tests/unit/test_llama_init.py
```

Exit 1, **1 failed, 41 passed in 8.55s**. The explicit case stops at
`TypeError: RunConfig.__init__() got an unexpected keyword argument 'attention_implementation'`.
The default case and existing regressions pass.

Next, the user adds the field and forwards explicit values to the model
setter. Rerun the regression to verify Green. The laptop CUDA smoke still
requires a compatible FA3 build with SM8x enabled.

## Verification after adding FA3 dispatch — 2026-11-20

The original CPU regression exits 0 with **42 passed in 4.42s**. The FA3
string reaches the setter in the agreed order, and omitting the field leaves
the default path alone. The agreed API also accepts other explicit backends.
Adding an explicit `"eager"` case gives **1 failed, 42 passed in 4.92s** with
the same command. The setter call is missing because `train()` only forwards
the exact `"flash_attention_3"` string.

The real single-GPU `train()` path confirmed the same mismatch. A tiny
Llama completed one BF16 update on the RTX 3080 Laptop after being configured
with `"eager"`, but Transformers still reported `"sdpa"`. The Python stdin
command used `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
with local GPU access and exited 1 on the explicit backend assertion.

Next, forward every non-`None` explicit backend and rerun the CPU regression
and real CUDA smoke. The FA3 kernel itself was not tested in this round.

## Backend selection Green — 2026-11-20

The user changed the guard to forward every non-`None` backend. The targeted
CPU regression command above exits 0 with **43 passed in 5.71s**. The real
`train()` smoke on the RTX 3080 Laptop completes one BF16 update per backend.
Transformers selects `"sdpa"` with the default `None` and `"eager"` when
explicitly requested. The Python stdin command used
`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -`
with local GPU access, exit 0.

Actual FA3 forward, backward, and throughput remain unverified. The FA3
extension is absent, and the available local build toolchain still has the
CUDA major-version mismatch recorded above.

## PR #17 wrapper follow-up — 2026-11-21

The [review](https://github.com/Aubadia/Upwash/pull/17#discussion_r4651836158)
reports that the Transformers FP32-to-BF16 cast warning causes graph breaks
with compiled FA3. Proposed first increment, pending agreement: select an
Upwash wrapper using `RunConfig.attention_implementation="upwash_flash_attention_3"`
and register that name through Transformers `AttentionInterface`. Start with
dense, unpadded causal Llama batches. Compare the cast, tensor layout and causal
scale at the native call boundary, and assert that the cast emits no warning.

Reference: Transformers **5.22.2** `flash_attention_forward` in
`transformers.integrations.flash_attention`, with PyTorch **2.15.0**. A probe
run via `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 .venv/bin/python -`
executed the real wrapper with a stand-in for the native call. It exited 0:
FP32 query and key and BF16 value reached the call as BF16 in
`[batch, sequence, heads, head_dim]` layout, with causal mode and the supplied
scale. The reference called `warning_once` once. This did not execute the
CUDA kernel.

```sh
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train_ddp.py tests/unit/test_train.py tests/unit/test_adaptor.py tests/unit/test_llama_init.py
```

Exit 0, **43 passed in 3.38s** for the existing routing and model tests.
Next: agree on the backend name and dense causal scope, then add and run the
first wrapper regression. No wrapper test or Red has been recorded yet.

### First Red with padding

The user asked to include padding in the first increment. The test uses the
proposed backend name and the existing Transformers 2D attention mask. It
checks a right-padded sequence with three valid tokens and a left-padded
sequence with two. The expected varlen inputs contain only valid tokens in
BF16, with `int32` cumulative offsets `[0, 3, 5]` for Q and K and maximum
length 3. The wrapper must retain causal mode and the attention scale, scatter
the output back into the original layout with zeros at padded positions, and
avoid the cast warning. This covers attention masking; Trainer batch format
and loss masking are outside the test.

```sh
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/test_adaptor.py tests/unit/test_llama_init.py
```

Exit 1, **1 failed, 43 passed in 2.75s**. The real Transformers 5.22.2
reference wrapper reached the FA3 varlen stand-in, restored the expected
padded layout, and emitted the cast warning. Upwash reaches the real
Transformers backend setter, which raises `ValueError` because
`upwash_flash_attention_3` is not registered. The Upwash mask and numerical
assertions have yet to run. Next: the user implements and registers the
backend and its mask handling, then rerun the regression. The native CUDA
kernel and padded-path compilation remain unverified.

### Registration review — 2026-11-21

Transformers 5.22.2 attempts to preload a FlashAttention kernel when the
registered name is `upwash_flash_attention_3`. The user chose `upwash_fa3`
for both registrations, and the regression now uses that name. The user's
`register_upwash_fa3_wrapper()` was verified directly using
`PYTHONPATH=src OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -`.
The probe exited 0: a tiny Llama selected the registered function, preserved
the mixed-padding 2D mask, and received `None` for the all-valid mask.

Reran the four-file regression command above after aligning the test name:
exit 1, **1 failed, 43 passed in 2.63s**. The reference passes, then
`train()` reaches the setter without having called the registration helper.
The setter rejects `upwash_fa3` with `ValueError`. Next: call the helper
before the setter on the opt-in path and implement the still-empty forward
callback. Only the test name and progress records were edited in this review.

### Dense first increment — 2026-11-21

The user agreed to limit the first increment to the fixed-length, unpadded
Llama batches used by current training and evaluation. Defer the varlen path
until a caller supplies padded batches. The regression now checks silent
BF16 conversion, native dense layout, GQA head counts, causal mode, attention
scale and the returned output. It runs the real Transformers 5.22.2 wrapper
against the dense kernel stand-in before importing the Upwash entry point.

```sh
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py -k dense_fa3
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_adaptor.py tests/unit/test_llama_init.py
```

The dense case exits 1 after the reference checks pass. `train.py` imports
`register_upwash_fa3_wrapper` from the kernels package, whose
`__init__.py` does not export it. A separate AST parse of `fa3.py` reports
`IndentationError` from the empty `upwash_fa3_forward()` body. These are
pre-execution blockers; the Upwash dense behavior has not run. The adaptor
and Llama tests exit 0 with **21 passed in 2.80s**. Next: the user fixes the
import and supplies a valid forward body, then rerun the dense case and
routing regressions. No production files were edited for this scope change.

### Dense wrapper implementation review — 2026-11-21

The forward body and registration call are present. The four-file targeted
regression command in the padding section exits 1 with **11 failed, 24 passed,
9 errors in 3.08s**. Every failure or setup error stops at the still-missing
`register_upwash_fa3_wrapper` export from the kernels package. The pinned
Transformers dense reference passes its assertions before that import.

A direct CPU probe using
`PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 .venv/bin/python -`
with a native-call stand-in reproduces three wrapper defects. CPU autocast
falls through because the autocast query omits the device, then the fallback
tries to read `q_prog` on the query tensor. Forcing the autocast branch
exposes the unsupported `softmax_scaling` keyword on dense inputs and an
unassigned `output` on masked inputs. Next: fix the import, use device-aware
autocast and the attention layer's `q_proj`, pass `softmax_scale`, and reject
a nonempty mask explicitly under the dense-only contract.

The Git FA3 source is configured as a required dependency without an explicit
revision. `uv.lock` has no FA3 entry, and `uv --no-cache pip show --python
.venv/bin/python flash-attn-3` reports the package absent. Real CUDA and
compiled execution remain unverified. Only progress records were updated.

### Follow-up verification — 2026-11-21

The direct module import reaches the registered wrapper through `train()`.
Rerunning the four-file targeted regression command exits 1 with **1 failed,
43 passed in 3.19s**. The Transformers reference passes. The Upwash dense
case now stops in the actual forward with `AttributeError` because CPU
autocast is queried without a device and the fallback reads `query.q_proj`
on a tensor. Next: call `torch.is_autocast_enabled(device)` and use
`model.q_proj.weight.dtype` in the fallback.

The direct CPU probe using the same Python stdin command forced the
autocast branch to isolate the other fixes. It exited 0 after verifying
BF16 layout, `softmax_scale=0.5`, causal mode, and explicit `ValueError`
for a non-`None` mask. Main-path dtype behavior is still Red. The package
is absent from the local environment, so real CUDA and compilation remain
unverified. Only progress records were updated.

### Autocast Green and projection fallback Red — 2026-11-21

The user changed the check to `torch.is_autocast_enabled(device)`. The
four-file targeted regression command exits 0 with **44 passed in 2.93s**,
including the CPU-autocast dense comparison. A standalone probe via
`PYTHONPATH=src OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -`
still raises `AttributeError` without autocast because the fallback reads
`query.q_proj` on a tensor.

Added a case to the dense regression with BF16 attention projection weights
and autocast disabled. The same targeted command now exits 1 with **1 failed,
44 passed in 2.68s**. Both reference runs pass. The new case reaches the
intended fallback and stops at `query.q_proj`. Next: read
`model.q_proj.weight.dtype` and recheck both cases. The local environment
still has no FA3 package; CUDA and compiled execution remain unverified.

### Dense CPU regression Green — 2026-11-21

The fallback now reads `model.q_proj.weight.dtype`. The four-file targeted
regression command in the padding section exits 0 with **45 passed in
3.43s**, including both dense cases with the native FA3 stand-in. The
reference comparison checks autocast and BF16 projection weights with
autocast disabled.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/integration/test_ddp_training.py tests/integration/test_runtime_process_group.py
```

The full CPU command inside the sandbox exits 1 with **406 passed and 5
failed in 4.27s** because the five multiprocessing tests could not bind a
localhost socket. The second command reruns those five cases with localhost
access and exits 0 with **5 passed in 19.48s**. All 411 CPU cases are
accounted for across the two runs.

FA3 is now a Linux-only optional dependency. The local environment and
`uv.lock` still do not contain it, and `uv --no-cache --offline lock --check`
exits 1 because the new Git source cannot be fetched offline. Next: verify
real BF16 FA3 forward, backward, and compiled graph breaks on CUDA. No
production or test files were edited during this verification.
