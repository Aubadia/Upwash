# 004 — Reject Empty Batches

## Behavior

Following the user's choice to reject empty batches and their implementation of a `ValueError` guard, `train(batches)` must raise `ValueError` when the supplied iterable yields no batches. Cover both an empty list and an empty iterator; model parameters must remain unchanged. Exception text, training mode after rejection, progress after errors, and returned-state snapshot semantics are not specified by this test. No API, dependency, or test configuration change is required.

## Verified Red — 2026-11-07

- `./test.sh tests/unit/core/test_trainer_empty_batches.py`: exit 1, `1 failed, 1 passed in 2.23s`.
- `./test.sh`: exit 1, `1 failed, 7 passed in 1.98s`.
- Empty list rejection passes. Empty iterator rejection fails with `DID NOT RAISE ValueError`.
- The current `if not batches` guard checks container truthiness. An empty iterator is truthy, so training returns normally. This is the intended Red; prior tests remain Green.

## Next Action

Task complete. Agree on the next behavior before adding tests; progress after errors and returned-state snapshot semantics remain deferred.

## Verified Green — 2026-11-07

- The user explicitly authorized the agent to modify `trainer.py` for this case.
- Removed the truthiness guard. After traversal, zero completed steps causes `ValueError("batches cannot be empty")`. Input is consumed once without materializing it or requiring a length.
- `./test.sh tests/unit/core/test_trainer_empty_batches.py`: exit 0, `2 passed in 2.03s`.
- `./test.sh`: exit 0, `8 passed in 1.91s`.
- Empty lists and iterators are rejected without parameter changes. Existing nonempty single-use iterator training scenarios remain Green. No dependencies or test configuration changed.
