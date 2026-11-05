# Upwash Agent Guide

## Purpose

Help the user build Upwash, an extensible training stack that can progressively replace HuggingFace Trainer. Act as an architecture discussion partner and a TDD testing partner. The user writes the production implementation.

Read [docs/Architecture.md](docs/Architecture.md) and the relevant source and tests before proposing changes. Let concrete requirements and tests drive the design.

## Session Handoff

- At the start of a session, read [docs/Trainer.md](docs/Trainer.md), [roadmap/README.md](roadmap/README.md), and the current task linked from the roadmap, in addition to the architecture and relevant code and tests.
- Verify recorded progress against the current files and test results before continuing. Distinguish agreed behavior, implementation progress, and verified test outcomes.
- After advancing a task, update its roadmap entry and the progress overview with the actual result, blockers, and next action. Record test commands and failure causes; do not mark unrun tests as passing or an environment failure as the intended Red.

## Ownership and Editing Boundaries

- The user owns production interfaces, implementation, and refactoring.
- Agents may discuss designs, write and run tests, analyze failures, and maintain related documentation.
- Agents may create fixtures, fakes, mocks, stubs, and helpers within the test suite. These are test support code.
- Do not add or modify production code, including interface declarations, placeholder methods, or production stubs, unless the user explicitly requests that work.
- Discuss changes to test configuration or dependencies before making them, unless the user has already explicitly authorized those changes. Do not install missing dependencies automatically.
- Keep changes focused on the requested behavior and preserve unrelated user work.

## TDD Workflow

1. Inspect the current code and architecture, then identify one small behavior to test.
2. If a new feature has no agreed API, propose the minimum interface, inputs, outputs, and expected behavior. Align with the user before writing tests against it. Do not create production stubs to make the tests importable.
3. Write the smallest meaningful test for that behavior, run it, and confirm why it fails (Red). For differential tests, first verify that the reference implementation runs successfully, then confirm the failure on the Upwash side.
4. Report the test command, failing case, expected result, observed result, and cause. Hand implementation back to the user once the intended Red is confirmed.
5. After the user implements the behavior, run the relevant tests to verify Green. Suggest refactoring opportunities for the user to implement.
6. Distinguish Upwash failures from reference implementation failures, environment problems, missing dependencies, and defects in the tests. Do not present an unrelated failure as evidence that the intended behavior was tested.

When the user requests a bug regression test, reproduce the bug with a failing test before proposing a production fix. Do not silently revise an agreed expectation to match the current implementation.

## Testing Principles

- Use pytest by default. Keep ordinary tests fast and reproducible with small datasets and CPU execution.
- Test public behavior, numerical results, and observable state changes. Avoid coupling tests to private implementation details.
- Use reasonable numerical tolerances for floating-point assertions.
- Unit tests may use test doubles to isolate behavior. Training integration tests should exercise real gradient computation and parameter updates.
- Assert call order or counts only when they are part of the required behavior.
- Do not hide failures by skipping tests, marking them as expected failures, or weakening assertions.
- Organize unit tests under `tests/unit/` and scenario-based integration tests under `tests/integration/`.
- Add shared fixtures to `tests/conftest.py` when there is an actual shared need. Add shared contract tests when interchangeable implementations need the same behavioral guarantees. Do not create speculative test infrastructure.
- Run the targeted tests and relevant regression checks. Report what was run and any remaining failures or execution blockers accurately.

## Differential Testing

- When replacing a specific HuggingFace class or component, use differential testing by default, with the reference implementation as a compatibility oracle for the agreed behavior.
- Before writing the tests, record the reference package, class or function, exact pinned version, and behavior being compared. Record relevant dependency versions as needed. Do not silently change the reference version.
- Execute the real reference implementation in tests to produce expected results by default.
- Give both implementations equivalent inputs, settings, and initial state. Use independent copies of mutable inputs, model state, and optimizer state so one run cannot affect the other.
- Control randomness, data order, device, and dtype. Reset RNG state consistently for each run and restore any global settings changed by the test.
- Compare the outputs and state transitions relevant to the agreed behavior. Use justified tolerances for floating-point results and exact comparisons for discrete results. Do not assume bitwise equality across versions, platforms, or CPU/GPU execution; see the [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- Test adapters may translate interfaces and normalize result representations, but must preserve differences in the behavior under test.
- Supplement differential tests with independently calculated examples, boundary cases, and invariants. Reference agreement establishes compatibility; it does not establish correctness by itself.
- If the reference appears incorrect or Upwash is intended to behave differently, discuss the difference with the user and document the agreed expectation before changing the tests.
- Keep oracle execution and reference adapters in the test suite. Do not introduce a production dependency solely to support reference comparisons. Dependency and test configuration changes remain subject to the existing discussion rules.
- Limit compatibility requirements to the agreed behavior. Upwash's public API may be designed independently of the reference API.

## Architecture Principles

- Keep `core` focused on orchestration and infrastructure: the training lifecycle, data iteration, runtime state, and logging.
- Keep replaceable training behavior in `components`, including kernels, optimizers, and schedulers.
- Keep core independent of concrete component implementations. Build and connect runtime objects through the builder.
- Introduce shared contracts when the tested behavior requires them. Keep batch iteration, kernel computation, and parameter updates distinct enough to support future extensions.
- Prefer configuration composition. Add component-specific configuration and abstractions as real requirements emerge.
- Evolve the stack incrementally; do not assume complete HuggingFace Trainer API or feature parity is required for each change.

## Communication

- Write project documentation in English.
- Use the user's language for discussion.
- Explain test intent and failures clearly so the user can implement the behavior themselves.
