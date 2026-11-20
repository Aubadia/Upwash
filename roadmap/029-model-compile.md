# 029 — Opt-in model compilation

Branch `feat/acceleration`. The model-only entry-point behavior follows
`origin/experiment` at `8718eb1`.

## Agreed behavior

- `RunConfig` accepts optional `compile_model=False` and
  `compile_mode="default"`. The default training path remains eager.
- When enabled, `train(config)` compiles the model with the selected mode
  and supplies the compiled callable to Trainer. `TrainResult.model` still
  exposes the original model. The first test uses the single-process path.

## First Red — 2026-11-20

The CPU unit test uses a compiler double that returns a distinct wrapper. It
checks that the wrapper reaches Trainer, that default and explicit modes are
forwarded, and that the eager path never calls the compiler. The Trainer double
isolates entry-point wiring. Actual compiler execution, numerical agreement,
and distributed compilation with the option enabled remain unverified.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ./test.sh -p no:cacheprovider --tb=short tests/unit/test_train.py tests/unit/test_train_ddp.py tests/unit/test_llama_init.py tests/unit/test_adaptor.py tests/unit/test_model_config.py
```

Exit 1, **2 failed, 92 passed in 4.51s**. The default-mode and explicit-mode
cases stop at `TypeError: RunConfig.__init__() got an unexpected keyword
argument 'compile_model'`. The eager case and existing regressions pass.

Next, the user implements the `RunConfig` fields and compile selection in
`train()`, then rerun the regression to verify Green.
