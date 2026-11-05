# Minimal Trainer Goal

This document records the agreed destination for the first training loop. Implementation progress and test evidence live in [the roadmap](../roadmap/README.md). Follow the responsibility boundaries in [Architecture.md](Architecture.md).

## Target Usage

```python
trainer = Trainer(
    model=model,
    kernel=kernel,
    optimizer=optimizer,
)

state = trainer.train(batches)
```

This is the agreed target API, not an implemented example. Build toward it one small, tested behavior at a time.

## Agreed Behavior

- `batches` supplies already assembled batches. The initial example may use a Python list.
- `kernel(model, batch)` returns a scalar loss tensor connected to the model's computation graph. For the first supervised kernel, `batch` is `(inputs, targets)` and the loss is `loss_fn(model(inputs), targets)`.
- The supervised kernel receives a required, externally supplied `loss_fn` when it is constructed or configured, with no default loss function. The supplied callable accepts predictions and targets and returns a scalar tensor; its configuration controls reduction. The Trainer receives the kernel and does not need a separate loss-function argument.
- `train(batches)` traverses the supplied iterable once and performs one optimizer update per batch.
- The trainer puts the model in training mode, clears gradients for each update, obtains the loss through the kernel, runs backward, and calls the optimizer to update parameters.
- The returned `state.global_step` counts successful optimizer updates in that call.

The initial execution scope is CPU, float32, one model, and one optimizer. Use a real `torch.optim.SGD` for the first update scenarios, without momentum or weight decay.

## Responsibilities

| Part | Initial responsibility |
| --- | --- |
| Trainer (`core/trainer.py`) | Coordinate batch iteration, training mode, gradient clearing, backward, parameter updates, and progress. |
| Kernel (`components/kernels/`) | Interpret a batch, call the model, and compute the scalar loss using the supplied `loss_fn`. |
| Optimizer | Convert gradients into parameter updates; initially inject an existing PyTorch optimizer. |
| State (`core/state.py`) | Expose the completed update count as `global_step`. |
| Builder (`builder.py`) | Construct and connect runtime objects when configuration-driven construction becomes necessary. Direct injection is sufficient for the initial tests. |

Core must not import a concrete kernel or optimizer implementation. Introduce shared contracts only as the tested interactions require them. The kernel's concrete public name, import path, and construction/configuration syntax for supplying `loss_fn` still need to be agreed before its first test is written; a class is not required.

## Completion Criteria

The milestone is complete when tests demonstrate that the target API:

1. Computes scalar loss through a real kernel using the externally supplied loss function, including cases with different loss functions.
2. Preserves the gradient path and produces the expected model gradients.
3. Performs a real parameter update with the expected numerical result.
4. Handles successive batches without unintentionally accumulating gradients.
5. Sets training mode and returns the correct completed update count.

For numerical test examples, use a bias-free model `prediction = w * x`, initialized with `w = 0`, and a batch with `x = 1` and `target = 2`. Supplying `torch.nn.MSELoss(reduction="mean")` gives initial loss `4` and gradient `-4`. With SGD learning rate `0.1`, the first update produces approximately `w = 0.4`; a second identical batch produces approximately `w = 0.72`. Supplying `torch.nn.L1Loss(reduction="mean")` for the initial state instead gives loss `2`, verifying that the kernel uses the supplied loss function. These are test configurations, not fixed kernel behavior. Use suitable floating-point tolerances.

## Deferred Decisions

Multiple epochs, device management, gradient accumulation, mixed precision, schedulers, logging, evaluation, checkpointing, and distributed training are outside this first milestone. Repeated-call state semantics and empty-input/error behavior need explicit agreement when their tasks are reached. The current API does not settle how future optimizers requiring repeated loss evaluation will interact with the loop.

No Hugging Face compatibility behavior or reference version has been selected. Before a differential comparison, agree on its scope and record the exact reference version according to AGENTS.md.

## References

- [PyTorch: Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html): the basic loss, backward, gradient clearing, and optimizer update flow.
- [Hugging Face: Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer): a reference for training responsibilities as Upwash grows.

