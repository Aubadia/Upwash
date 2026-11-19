"""Trainer token metrics report global counts under data parallelism (task 026).

Each of `world_size` replicas consumes an equally shaped batch per update, so the
global token count is the local count times world_size. The lab score
(oj_total_tokens) is a global token count. world_size is bookkeeping only: DDP has
already averaged the gradients, so the update itself must not change.
"""
import time

import pytest
import torch

from upwash.builder import build_trainer
from upwash.config import TrainingConfig
from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer


def make_trainer(world_size=None, *, pad_token_id=None, logging_steps=2):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = {} if world_size is None else {"world_size": world_size}
    trainer = Trainer(model=model, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optimizer,
                      logger=MemoryLogger(), logging_steps=logging_steps, count_tokens=True,
                      pad_token_id=pad_token_id, **kwargs)
    return trainer, model, optimizer


def batches(pattern, sizes=(8, 8, 4)):
    for n in sizes:
        yield (torch.tensor(pattern * (n // len(pattern)), dtype=torch.int64),
               torch.zeros(n, dtype=torch.int64))


def metric(trainer, key):
    return [r["metrics"][key] for r in trainer.logger.records]


@pytest.mark.parametrize(("pad_token_id", "pattern", "local_totals"),
                         [(None, [1, 1], [16, 20]), (0, [0, 1], [8, 10])],
                         ids=["no-padding", "padding-excluded"])
def test_total_tokens_seen_is_the_local_count_times_world_size(pad_token_id, pattern, local_totals):
    trainer, _, _ = make_trainer(4, pad_token_id=pad_token_id)

    trainer.train(batches(pattern))

    totals = metric(trainer, "train/total_tokens_seen")
    assert all(type(t) is int for t in totals)
    assert totals == [4 * t for t in local_totals]


def test_tokens_per_second_is_the_global_rate(monkeypatch):
    elapsed = [0.0]
    trainer, model, optimizer = make_trainer(4)
    # Each update costs 1s loading + 1s forward + 1s optimizer step on the simulated clock.
    model.register_forward_pre_hook(lambda module, args: elapsed.__setitem__(0, elapsed[0] + 1))
    optimizer.register_step_post_hook(lambda o, a, k: elapsed.__setitem__(0, elapsed[0] + 1))

    def timed():
        for batch in batches([1, 1]):
            elapsed[0] += 1
            yield batch

    monkeypatch.setattr(time, "perf_counter", lambda: elapsed[0])
    trainer.train(timed())

    # Interval 1: 2 x 8 local tokens in 6 s; final interval: 4 local tokens in 3 s.
    assert metric(trainer, "train/tokens_per_second") == pytest.approx([4 * 16 / 6, 4 * 4 / 3])


def test_world_size_one_is_the_default():
    default, _, _ = make_trainer()
    explicit, _, _ = make_trainer(1)

    default.train(batches([1, 1]))
    explicit.train(batches([1, 1]))

    assert metric(default, "train/total_tokens_seen") == metric(explicit, "train/total_tokens_seen") == [16, 20]


def test_world_size_does_not_change_the_update_or_the_loss():
    single, single_model, _ = make_trainer(1)
    replica, replica_model, _ = make_trainer(4)

    single.train(batches([0, 1]))
    replica.train(batches([0, 1]))

    torch.testing.assert_close(replica_model.weight, single_model.weight, rtol=0, atol=0)
    assert metric(replica, "train/loss") == metric(single, "train/loss")
    assert single_model.weight.abs().sum() > 0  # the comparison is not between untrained models


@pytest.mark.parametrize("world_size", [0, -2])
def test_a_non_positive_world_size_is_rejected(world_size):
    with pytest.raises(ValueError, match="world_size"):
        make_trainer(world_size)


def test_build_trainer_forwards_world_size(make_optimizer_config, sgd_optimizer_factory):
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Embedding(2, 2)
    logger = MemoryLogger()
    trainer = build_trainer(config=TrainingConfig(optimizer=make_optimizer_config(lr=0.1), logging_steps=1),
                            model=model, loss_fn=torch.nn.CrossEntropyLoss(),
                            optimizer_factory=sgd_optimizer_factory, logger=logger,
                            count_tokens=True, world_size=8)

    trainer.train([(torch.ones(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64))])

    assert logger.records[0]["metrics"]["train/total_tokens_seen"] == 32
