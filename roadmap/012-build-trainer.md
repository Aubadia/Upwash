# 012 — Build a Configured Trainer

## Status and Dependencies

Planned after [009](009-training-config.md) and [011](011-training-loss-logging.md). Construction API remains provisional until those contracts are verified and the user agrees.

## Proposed Minimal Construction Path

Start with a function rather than requiring a stateful Builder class:

```python
from upwash.builder import build_trainer

trainer = build_trainer(
    config=config,
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=torch.optim.SGD,
    logger=logger,
)
state = trainer.train(loader)
```

Inputs are an explicit `TrainingConfig`, existing model and loss function, an optimizer factory accepting model parameters and keyword `lr`, and an optional logger. The builder constructs the optimizer using the supplied model's parameters and configured learning rate, then wires the Trainer. It returns a Trainer without executing training. Model/loss creation and data loader construction stay with the caller for this first scenario.

An injected factory keeps optimizer selection explicit and avoids a speculative string registry or fixed default optimizer. Core receives the constructed optimizer and logger; it does not import concrete optimizer implementations or the builder/config to construct them itself.

## TDD Sequence

1. Agree on the construction interface; confirm whether a function is sufficient for the actual caller.
2. Test the public result with real CPU SGD: configured `lr=0.1`, initial weight `0`, input `1`, target `2`. The built trainer updates the supplied model to approximately `0.4`, returns step `1`, and exposes loss `4.0` through the supplied logger.
3. Use a second learning rate, such as `0.05`, to require a weight of approximately `0.2`; this detects ignored configuration.
4. Verify omitted logging remains usable and construction itself leaves model parameters unchanged. Run the full regression suite.

Use real parameter updates to detect wiring an optimizer to the wrong parameters. Broader model factories, loader construction, schedulers, plugins, and complete run orchestration can follow demonstrated requirements.

No tests run for this feature; no implementation or Red claimed.
