# 017 — Optimizer and Schedule Configuration for the Sweep

## Behavior

The experiment plan trains Muon (hidden matrices) plus AdamW (embeddings, LM head) and sweeps nine hyperparameters at 150M. Every swept value must be a configuration field. This task defines the fields and their validation. It doesn't compute schedules or build optimizers.

Branch `feat/optimizer-schedule-config`, created from `main` at `1d2dc74` (after PR #6).

## Experiment Plan (from the user, 2026-11-12)

| Grant per run | Scale | Tokens | Grants |
| --- | --- | --- | --- |
| 40 H200-hours | 150M | 30B | ~60 |
| 110 H200-hours | 300M | 60B | 8 |
| 266 H200-hours | 500M | 100B | 3 (1 before calibration, 4 more after) |
| 540 H200-hours | 750M | 150B | 1 |

Sweep order is top to bottom. Values in the same grid group are searched jointly:

- `lr`: 0.002, 0.004, 0.008, 0.016, 0.032 (group 1)
- Muon beta: 0.9, 0.95, 0.98, 0.99 (group 1)
- AdamW beta1: 0.85, 0.9
- AdamW beta2: 0.95, 0.98, 0.99 (group 2)
- Adam LR multiplier: 1.5, 1.6, 1.7 (group 2)
- Warmup ratio: 0.001, 0.003, 0.01
- Decay ratio: 0.1, 0.2, 0.3, 0.4, 0.5
- Decay shape: linear, cosine, `(1-x)^p` with p in 1.0, 1.5, 2.0
- Weight decay: 0.04, 0.08, 0.16, 0.32, 0.64, divided by `d` (hidden dimension)

Initialization is fixed, not swept:

- Hidden matrices: std `1/sqrt(d_in)`.
- LM head: std `1/d`.
- Token embedding: std 0.1.

These rules belong in the model's initialization, not in configuration.

The sweep is about 48 runs at 150M: 20 + 2 + 9 + 3 + 5 + 4 + 5. Linear decay is `p = 1.0`, so the decay shape adds 4 runs, not 5. About 60 grants are available.

## Decisions — 2026-11-12

- **Starting point:** the first value of every grid. It lives in the sweep's base config, not as class defaults: lr 0.002, Muon beta 0.9, beta1 0.85, beta2 0.95, multiplier 1.5, warmup 0.001, decay 0.1, power decay with `p = 1.0` (linear), weight decay 0.04. The agent noted that edge values can bias the lr choice, and suggested spending spare grants to recheck lr at the end.
- **Decay to zero.** There is no `min_lr` field. All planned shapes reach zero at the end of decay. The agent cited Bergsma et al. 2025, "Straight to Zero" (arXiv 2502.15938), and the WSD schedules of Hägele et al. 2024.
- **Muon uses `adjust_lr_fn="match_rms_adamw"`.** This matches the plan's grid, where AdamW lr is about 1.6 × Muon lr and both share a `c/d` weight decay. It is fixed in the future optimizer builder, not a config field. Whether it matches the teammates' 150M baseline still needs checking. Torch 2.15.0 in the lockfile provides `torch.optim.Muon(lr, weight_decay, momentum, nesterov, adjust_lr_fn, ...)`.
- **Every field is required; there are no defaults.** A sweep config that forgets a value fails at construction.
- **The builder passes only `lr` to `optimizer_factory`.** Task 012 forwarded `**config.optimizer.to_dict()`, which would hand Muon/AdamW fields to SGD. Building Muon plus AdamW with parameter groups is task 019.
- **`ScheduleConfig` stays standalone.** It isn't composed into `TrainingConfig` until a top-level experiment config needs it.

## Agreed Interface

```python
from upwash.config import OptimizerConfig, ScheduleConfig, TrainingConfig

optimizer = OptimizerConfig(
    lr=0.002,                 # Muon learning rate
    muon_beta=0.9,
    adam_beta1=0.85,
    adam_beta2=0.95,
    adam_lr_multiplier=1.5,   # AdamW lr = lr * multiplier
    weight_decay=0.04,        # coefficient c; applied as c / d_model
)
schedule = ScheduleConfig(
    warmup_ratio=0.001,
    decay_ratio=0.1,
    decay_shape="power",      # "cosine" or "power"; linear is power with decay_power=1.0
    decay_power=1.0,
)
config = TrainingConfig(optimizer=optimizer)
```

| Field | Accepted |
| --- | --- |
| `lr` | finite, `>= 0` (task 010) |
| `muon_beta`, `adam_beta1`, `adam_beta2` | `[0, 1)` |
| `adam_lr_multiplier` | finite, `> 0` |
| `weight_decay` | finite, `>= 0` |
| `warmup_ratio`, `decay_ratio` | `[0, 1]` each, sum `<= 1` |
| `decay_shape` | `"cosine"` or `"power"` |
| `decay_power` | finite, `> 0` |

Invalid values raise `ValueError`. A missing field raises `TypeError` naming it. `to_dict()` returns every field. `decay_power` is required and validated even for cosine, where it is unused.

Deferred:

- Schedule computation and the Muon/AdamW builder: task 019 and later.
- `DataConfig` (task 018), `TrainingConfig` run settings (batch tokens, device, precision, `log_every`), `ModelConfig`, and a top-level experiment config.
- Non-float input types.

Open questions for task 019 and later:

- Batch tokens per scale.
- Which parameters receive weight decay.
- Tied or untied embeddings.
- The optimizer for RMSNorm gains.
- Whether lr transfers across widths.

## Tests

- `tests/conftest.py`: the `sweep_start_optimizer` fixture holds the starting values. The `make_optimizer_config(**overrides)` fixture builds a valid config, so tests override only the field they test. The config import is inside the fixture, so a broken config module doesn't stop collection of unrelated tests.
- `tests/unit/test_training_config.py`:
  - The four task 010 cases now build full configs.
  - New cases cover every field exposed, each missing field raising `TypeError` with its name, and betas accepted at 0, 0.5, and 0.99.
  - Betas are rejected at -0.1, 1.0, 1.5, and NaN. The multiplier is rejected at 0, -1, NaN, and infinity.
  - Zero weight decay is accepted. Negative, NaN, and infinite weight decay are rejected.
  - `to_dict()` returns all six fields.
- `tests/unit/test_schedule_config.py`:
  - Fields exposed, and each missing field raises `TypeError`.
  - Accepted ratios: 0/0, 1/0, 0/1, 0.5/0.5, and 0.01/0.5. Rejected ratios: -0.1, 1.1, and NaN, and warmup 0.6 with decay 0.5 (sum above 1).
  - `"cosine"` and `"power"` are accepted. `"linear"`, `"Cosine"`, `"step"`, and `""` are rejected.
  - Powers 1.0, 1.5, and 2.0 are accepted. 0, -1, NaN, and infinity are rejected.
  - `to_dict()` returns all four fields.
- `tests/unit/test_builder.py` and `tests/integration/test_built_trainer.py` build full configs and still expect SGD with the configured `lr`.

## Verified Red — 2026-11-12

Command (Docker, `upwash-venv-pr5`, which includes wandb from PR #5):

```sh
PYTHONPATH=src uv run pytest -q tests/unit/test_training_config.py tests/unit/test_builder.py tests/integration/test_built_trainer.py
```

Result: `47 failed`. The causes are missing API and behavior only:

- 41 cases: `TypeError: OptimizerConfig.__init__() got an unexpected keyword argument 'muon_beta'`. This covers the four task 010 cases, the two builder cases, and the two integration cases.
- 6 missing-field cases: the regex didn't match, because the current class rejects an extra argument instead of naming the missing one.

`tests/unit/test_schedule_config.py`: collection error, `ImportError: cannot import name 'ScheduleConfig'`.

The rest of the suite, ignoring that file, gives `47 failed, 82 passed`. No other test is affected.

Test sanity check: a scratch reference implementation outside the repository passed all four files (`78 passed`). It used dataclasses with `__post_init__` validation and a builder passing only `lr`, loaded as a pytest plugin.

## Verified Green — 2026-11-12

The user implemented the following:

- `OptimizerConfig` and `ScheduleConfig` are dataclasses. Validation is in `__post_init__`, and `to_dict()` returns `asdict(self)`.
- The old hand-written `OptimizerConfig` was removed.
- `build_trainer` calls `optimizer_factory(model.parameters(), lr=config.optimizer.lr)`.

Command (Docker, `upwash-venv-pr5`): the four files above returned `78 passed`. Full suite: `160 passed`.

Red iterations during implementation:

- The first version returned `48 failed, 112 passed`. The old `class OptimizerConfig` was still defined below the new dataclass, and the later definition shadowed the new one.
- An agent-run preview on a scratch copy without the old class showed `6 failed` from three causes:
  - `weight_decay > 0` rejected zero.
  - `decay_power >= 0` accepted zero.
  - The builder still forwarded every field to SGD.

Review follow-up: a `TrainingConfig.to_dict()` without tests had returned `{"optimizer": OptimizerConfig(...)}`, an object instead of a nested dict. The user removed it. Recording configs with `FlockLogger(config=...)` will be designed with the top-level experiment config. The user also fixed the `decay_shape` annotation (`str`) and the error messages. Full suite after the fixes: `160 passed`.

Status: complete for optimizer and schedule fields and validation. Committed as `7a8ed2c`. Next: task 018 (`DataConfig`) in the same pull request. Task 019 builds Muon plus AdamW with parameter groups from `OptimizerConfig`.
