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
```

Production modules live under `src/upwash/`. Module paths below are relative to that package directory. Tests and documentation remain at the repository root.

## Responsibilities and Dependencies

- `core`: Training lifecycle, data iteration, runtime state, and logging services.
- `components/kernels`: Model calls, batch interpretation, and loss and metric computation.
- `components/optimizers`: Rules for converting gradients into parameter updates.
- `components/schedulers`: Learning rate adjustments based on progress or evaluation results.
- `config.py`: Settings for a training run, organized through composition. Add component-specific configuration as needed.
- `builder.py`: Component construction and wiring.

Core uses shared contracts, which concrete components implement. The builder injects those components, keeping core independent of concrete kernels, optimizers, and schedulers. Initially, contracts shared between core and components live in `core/contracts.py`. Add shared implementations within each component family as needed.

Batch iteration, kernel computation, and parameter updates need clear interaction points to accommodate gradient accumulation and update methods that require repeated loss computation. Tests for these behaviors drive the concrete interfaces.

## Test Layers

| Layer | What It Verifies | Principles |
| --- | --- | --- |
| Unit | Computation, state transitions, and error handling within individual components | Use small datasets; inject simple fakes for flow tests |
| Contracts | Whether interchangeable implementations follow the same input, output, and state semantics | Add shared tests when alternative implementations or public extension interfaces are introduced |
| Integration | Cooperation between the trainer, data, kernel, and a real optimizer | Use small CPU models to verify numerical results and actual parameter updates |

Organize unit tests by module and name integration tests by scenario. Avoid mechanically creating a matching test file for every source file. Keep only fixtures with an established shared use in `conftest.py`.

Focus tests on public behavior, observable state, and numerical results. Assert call order or counts only when they are requirements themselves. Integration tests use real gradient computation and parameter updates so that the training flow is exercised beyond mocks.
