# Minimal Trainer Goal

This document records the destination for the first training loop. Implementation progress and test evidence live in [the roadmap](../roadmap/README.md). Follow the responsibility boundaries in [Architecture.md](Architecture.md).

## Kernel Terminology and API Status

In Upwash, kernels are operation implementations such as matmul, attention, and fused tensor operations. They are used at the operation layer within models or loss functions.

The earlier proposal to pass a training-computation `kernel` into `Trainer` has been withdrawn following this terminology clarification. A kernel does not interpret an entire training batch and invoke the model to produce a loss. Do not implement or test the earlier `kernel(model, batch)` interface.

The intended training entry point remains:

```python
state = trainer.train(batches)
```

The agreed interface for task 001 is:

```python
from upwash.core.trainer import Trainer

trainer = Trainer(model=model, loss_fn=loss_fn)
loss = trainer.compute_loss(inputs, targets)
```

Trainer holds the model and externally supplied loss function. `compute_loss` calls `model(inputs)` and returns `loss_fn(predictions, targets)` as a scalar tensor, preserving its gradient path. This is the first supervised calling convention. The first training loop will unpack `(inputs, targets)` batches; other formats are deferred until required. Optimizer injection is agreed in [task 002](../roadmap/002-minimal-training-loop.md): `Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)` followed by `state = trainer.train(batches)`. Following the user’s correction, the optimizer argument is required, including for loss-only tests. Do not introduce a replacement training-step abstraction merely to preserve the previous constructor shape.

## Agreed Behavior

- `batches` supplies already assembled batches. The initial example may use a Python list.
- Batch contents depend on the task. The first supervised flow supports `(inputs, targets)` batches interpreted by Trainer; this is not a universal batch format or operation-kernel contract.
- The loss function is externally supplied to the Trainer constructor, with no fixed MSE assumption or default loss function.
- For the first supervised numerical example, forward/loss computation produces `loss_fn(model(inputs), targets)`, a scalar tensor connected to the model's computation graph. The supplied loss function controls reduction.
- `train(batches)` traverses the supplied iterable once and performs one optimizer update per batch.
- The training flow sets training mode, clears gradients for each update, computes loss, runs backward, and calls the optimizer to update parameters.
- The returned `state.global_step` counts successful optimizer updates in that call.

The initial execution scope is CPU, float32, one model, and one optimizer. Use a real `torch.optim.SGD` for the first update scenarios, without momentum or weight decay. Existing PyTorch operations are sufficient; custom operation kernels are not a prerequisite for this milestone.

## Responsibilities

| Part | Initial responsibility |
| --- | --- |
| Trainer (`core/trainer.py`) | Hold the model and loss function; expose `compute_loss(inputs, targets)`. Later coordinate batch iteration, training mode, gradient clearing, backward, parameter updates, and progress. |
| Model and loss function | Produce predictions through `model(inputs)` and calculate scalar loss through `loss_fn(predictions, targets)`. |
| Kernels (`components/kernels/`) | Implement individual operations such as matmul and attention when a replacement is needed. |
| Optimizer | Convert gradients into parameter updates; initially inject an existing PyTorch optimizer. |
| State (`core/state.py`) | Expose the completed update count as `global_step`. |
| Builder (`builder.py`) | Construct and connect runtime objects when configuration-driven construction becomes necessary. |

Core must not import concrete component implementations. Introduce shared contracts only as the tested interactions require them.

## Completion Criteria

The milestone is complete when tests demonstrate that the training flow:

1. Computes scalar loss using the externally supplied loss function, including cases with different loss functions.
2. Preserves the gradient path and produces the expected model gradients.
3. Performs a real parameter update with the expected numerical result.
4. Handles successive batches without unintentionally accumulating gradients.
5. Sets training mode and returns the correct completed update count.

For numerical test examples, use a bias-free model `prediction = w * x`, initialized with `w = 0`, and input `x = 1` with target `2`. Supplying `torch.nn.MSELoss(reduction="mean")` gives initial loss `4` and gradient `-4`. With SGD learning rate `0.1`, the first update produces approximately `w = 0.4`; a second identical batch produces approximately `w = 0.72`. Supplying `torch.nn.L1Loss(reduction="mean")` for the initial state instead gives loss `2`. These are test configurations, not fixed training behavior. Use suitable floating-point tolerances.

## Deferred Decisions

Multiple epochs, device management, gradient accumulation, mixed precision, schedulers, logging, evaluation, checkpointing, distributed training, and custom operation kernels are outside this first milestone. Repeated calls count completed updates separately (`1 → 1` for two one-batch calls), while model parameters continue from their trained values. Returned-state snapshot semantics and empty-input/error behavior need explicit agreement when their tasks are reached. Future optimizers requiring repeated loss evaluation may require different interactions.

No Hugging Face compatibility behavior or reference version has been selected. Before a differential comparison, agree on its scope and record the exact reference version according to AGENTS.md.

## Subsequent Evaluation Extension

The later [small evaluation extension](../roadmap/017-small-evaluation.md) adds
`result = trainer.evaluate(batches)` with `loss` (Python float) and `num_targets`
(Python int). For the initial dense, unpadded scope, the supplied scalar loss
must be a mean over target elements, and evaluation weights each batch by
`targets.numel()`. It uses eval mode/no-grad, restores the previous uniform mode
on success or a loss exception, rejects empty input, and preserves existing
gradients, parameters, optimizer state, and training progress. These behaviors
and real tiny-LM training/evaluation are verified Green on 2026-11-13. Real dev
subset preparation and final Flock scoring remain
pending. This extension does not change the original training milestone above.

## Scheduled Evaluation

`TrainingConfig(eval_steps=N)` wires the interval through Builder to Trainer.
`trainer.train(batches, eval_batches=dev_loader)` evaluates after every N
completed updates and once at normal completion unless that step was already
evaluated. The dev iterable must support repeated traversal; initially use
dense, unpadded targets with a mean-reduced supplied loss, as in `evaluate()`.
Scheduling is inactive when the interval or dev iterable is omitted.

Evaluation runs independently of training logging and still runs without a
logger. With a logger it emits `eval/loss`, combining coincident training and
evaluation metrics in one event, including the final step. A step's metrics
are retained until the next batch is fetched or training finishes, allowing
final evaluation to join an event at an exact training-log boundary.
Evaluation and logging time are excluded from training throughput, including
when the logging and evaluation intervals differ. Evaluation preserves model
mode and training gradients through the existing evaluator. Steps remain
per-call; this increment does not add perplexity, validate interval values,
or make a one-shot dev iterator reusable.

## Token Counting Extension

Trainer and `build_trainer` accept optional `count_tokens=False` and
`pad_token_id=None`. With `count_tokens=True`, completed training updates add
the number of input elements, excluding values equal to a configured padding
id. Without a padding id, zero is a valid token. Targets are not counted.

The total persists across `train()` calls on the same Trainer, including calls
without a logger, and excludes `evaluate()`. Enabled interval and final-partial
events include `train/total_tokens_seen` as a Python integer. Omitted or false
`count_tokens` preserves generic training without token metrics. Padding
exclusion changes this metric only; the supplied loss controls loss masking.
Log steps retain their existing per-call semantics. These CPU behaviors are
verified; GPU/distributed counting and checkpoint
persistence of the counter are not established.

## Throughput Logging

With token counting and interval logging enabled, `train/tokens_per_second`
reports each interval's non-padding input tokens divided by elapsed
`time.perf_counter()` seconds. Timing starts before fetching the first batch
and stops after the final update and token count in that interval. Data loading,
forward, backward, and optimizer work are included; metric emission is excluded.
The next interval starts after logging. A final partial interval uses the last
update's timestamp, excluding the time spent discovering iterator exhaustion.

Each `train()` call starts fresh interval tokens and timing, while
`train/total_tokens_seen` continues accumulating. Evaluation and gaps between
training calls do not affect the next interval's rate. Zero elapsed time emits
`0.0`; all-padding intervals also emit `0.0`. Disabling token counting omits
throughput. These behaviors are verified with a simulated clock and real CPU
training; GPU synchronization and distributed throughput remain unverified.

## References

- [PyTorch: Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html): the basic loss, backward, gradient clearing, and optimizer update flow.
- [Hugging Face: Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer): a reference for training responsibilities as Upwash grows.
