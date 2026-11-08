# 006 — Fit a Small Linear Regression End to End

## Behavior

Verify that the existing minimal framework learns a real, if tiny, problem: `TensorDataset`, `DataLoader`, `Trainer`, a real `torch.nn.Linear(1, 1)` with bias, `torch.nn.MSELoss(reduction="mean")`, and `torch.optim.SGD(lr=0.1)` working together. This is a verification scenario for the minimum viable framework, not a new feature. No new production API was required.

## Scenario

- Data: 16 noise-free samples, `inputs = linspace(-1, 1, 16)` reshaped to `(16, 1)`, `targets = 3 * inputs + 1`.
- Loader: `batch_size=4`, dataset order, so each `train(loader)` call is one epoch of 4 updates.
- Model: weight and bias initialized to zero; RNG restored after initialization. CPU float32.
- Training: 30 successive `trainer.train(loader)` calls; each returns `global_step == 4` (per-call progress, as agreed in task 002).
- Evaluation: full-data loss via `trainer.compute_loss` under `torch.no_grad()`.

Assertions: initial loss above `4.0`; final loss below `1e-4`; weight within `1e-3` of `3` and bias within `1e-3` of `1`.

Threshold justification: a standalone PyTorch run of the same setup gave initial loss approximately `4.4`, final loss approximately `2e-9`, weight approximately `2.99993`, and bias approximately `1.00001` after 30 epochs (approximately `0.0034` loss and weight `2.909` after 10 epochs). The tolerances leave a wide margin while still requiring convergence.

## Verification — 2026-11-08

- Test: `tests/integration/test_fit_linear_regression.py::test_repeated_training_fits_a_small_linear_regression`.
- Command (Docker): `PYTHONPATH=src uv run pytest -q tests/integration/test_fit_linear_regression.py` returned `1 passed`; the full suite returned `19 passed`.
- No Red is claimed: the test passed on its first run against existing code, as expected for a verification scenario.
- Sensitivity check: a scratch copy with `lr=0.0` failed with `assert 4.399999618530273 < 0.0001`, confirming the assertions detect training that does not learn. The repository test was not changed.

## Remaining Gaps for the Minimum Viable Framework

- Training loss is not observable from `train()`; only `global_step` is returned. This belongs to the Logger area.
- Shuffle, multiple epochs within one `train()` call, and invalid `batch_size` handling remain deferred and are not required for this scenario.
