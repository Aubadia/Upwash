# Upwash Architecture

Upwash is an extensible training framework. Start with a minimal working training flow, and let actual requirements and tests drive the evolution of its interfaces. Add configuration, component types, and abstractions as needed.

## Current Directory Structure

The following tree reflects the current local files and directories, excluding Git metadata and generated caches.

```text
Upwash/
├── .gitignore
├── .python-version
├── README.md
├── AGENTS.md
├── docs/
│   ├── Architecture.md
│   └── Trainer.md
├── roadmap/
│   ├── README.md
│   └── 001-kernel-loss.md
├── pyproject.toml
├── test.sh
├── src/
│   └── upwash/
│       ├── __init__.py
│       ├── builder.py
│       ├── components/
│       │   ├── kernels/
│       │   ├── optimizers/
│       │   └── schedulers/
│       ├── config.py
│       └── core/
│           ├── contracts.py
│           ├── data.py
│           ├── logger.py
│           ├── state.py
│           └── trainer.py
└── tests/
    ├── conftest.py
    ├── integration/
    └── unit/
        ├── components/
        └── core/
            └── test_trainer_loss.py
```

Production modules live under `src/upwash/`. Module paths below are relative to that package directory. Tests and documentation remain at the repository root.

## Responsibilities and Dependencies

- `core`: Training lifecycle, data iteration, runtime state, and logging services.
- `components/kernels`: Replaceable operation implementations such as matmul, attention, and fused tensor operations. Kernel interfaces operate on the tensors and parameters required by their operations; they do not represent a complete model-and-batch training computation.
- `components/optimizers`: Rules for converting gradients into parameter updates.
- `components/schedulers`: Learning rate adjustments based on progress or evaluation results.
- `config.py`: Settings for a training run, organized through composition. Add component-specific configuration as needed.
- `builder.py`: Component construction and wiring.

Core uses shared contracts where tested interactions require them, keeping it independent of concrete component implementations. The builder constructs and connects runtime objects as needed. Operation kernels belong at the operation layer used by models or loss functions; the minimal Trainer does not require a generic operation-kernel argument. Kernel selection and replacement mechanisms remain to be designed when a concrete operation requirement arises. Initially, contracts shared between core and components live in `core/contracts.py`. Add shared implementations within each component family as needed.

Batch iteration, forward/loss computation, and parameter updates need clear interaction points to accommodate gradient accumulation and update methods that require repeated loss computation. The agreed first supervised interface is `Trainer(model=model, loss_fn=loss_fn)` with `trainer.compute_loss(inputs, targets)`. Trainer holds the externally supplied model and loss function, calls `model(inputs)`, and returns `loss_fn(predictions, targets)` as a scalar tensor. The first training loop will interpret each batch as `(inputs, targets)`; this is the initial supported format, not a universal batch format. Additional calling conventions will be designed from concrete requirements. Task 002 now agrees on an externally constructed optimizer supplied through `Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)` and `state = trainer.train(batches)`. Following the user’s correction, optimizer injection is required, including when constructing Trainer for loss tests. The returned object exposes `global_step`.

## Test Layers

| Layer | What It Verifies | Principles |
| --- | --- | --- |
| Unit | Computation, state transitions, and error handling within individual components | Use small datasets; inject simple fakes for flow tests |
| Contracts | Whether interchangeable implementations follow the same input, output, and state semantics | Add shared tests when alternative implementations or public extension interfaces are introduced |
| Integration | Cooperation between the trainer, data, model/loss computation, and a real optimizer | Use small CPU models to verify numerical results and actual parameter updates |

Organize unit tests by module and name integration tests by scenario. Avoid mechanically creating a matching test file for every source file. Keep only fixtures with an established shared use in `conftest.py`.

Focus tests on public behavior, observable state, and numerical results. Assert call order or counts only when they are requirements themselves. Integration tests use real gradient computation and parameter updates so that the training flow is exercised beyond mocks.
