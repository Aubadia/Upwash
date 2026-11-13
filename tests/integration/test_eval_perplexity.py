import math

import pytest
import torch
import torch.nn.functional as F

from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer

# eval/perplexity = exp(eval/loss), valid only for a mean natural-log cross-entropy,
# so the caller opts in. A model whose logits are all zero predicts the uniform
# distribution over V tokens: cross-entropy ln V and perplexity exactly V. With
# lr=0 the real SGD updates leave it uniform, giving an independent expected value.

VOCAB = 8


def next_token_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def uniform_model():
    model = torch.nn.Embedding(VOCAB, VOCAB)
    with torch.no_grad():
        model.weight.zero_()
    return model


def token_batches(count):
    generator = torch.Generator().manual_seed(0)
    return [
        (torch.randint(0, VOCAB, (2, 4), generator=generator), torch.randint(0, VOCAB, (2, 4), generator=generator))
        for _ in range(count)
    ]


def run(loss_fn=next_token_loss, steps=3, **kwargs):
    model = uniform_model()
    logger = MemoryLogger()
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        logger=logger,
        eval_steps=2,
        **kwargs,
    )
    state = trainer.train(token_batches(steps), eval_batches=token_batches(2))
    return state, logger.records


def test_eval_perplexity_is_logged_beside_scheduled_and_final_eval_loss():
    state, records = run(eval_perplexity=True)

    assert state.global_step == 3
    # Scheduled evaluation at step 2, final evaluation at step 3.
    assert [record["step"] for record in records] == [2, 3]
    for record in records:
        metrics = record["metrics"]
        assert metrics["eval/loss"] == pytest.approx(math.log(VOCAB), rel=1e-6)
        assert isinstance(metrics["eval/perplexity"], float)
        assert metrics["eval/perplexity"] == pytest.approx(VOCAB, rel=1e-6)
        assert metrics["eval/perplexity"] == pytest.approx(math.exp(metrics["eval/loss"]), rel=1e-12)


@pytest.mark.parametrize("kwargs", [{}, {"eval_perplexity": False}], ids=["default", "explicit-false"])
def test_eval_perplexity_is_not_logged_unless_requested(kwargs):
    _, records = run(**kwargs)

    assert [record["step"] for record in records] == [2, 3]
    assert all("eval/loss" in record["metrics"] for record in records)
    assert all("eval/perplexity" not in record["metrics"] for record in records)


def test_eval_perplexity_overflow_is_logged_as_infinity_without_stopping_training():
    # A diverged sweep run can reach losses whose exp overflows a float (~709.78).
    def diverged_loss(logits, targets):
        return next_token_loss(logits, targets) + 1000.0

    state, records = run(loss_fn=diverged_loss, eval_perplexity=True)

    assert state.global_step == 3
    assert [record["step"] for record in records] == [2, 3]
    for record in records:
        assert record["metrics"]["eval/loss"] == pytest.approx(1000.0 + math.log(VOCAB), rel=1e-6)
        assert record["metrics"]["eval/perplexity"] == math.inf


def test_eval_perplexity_is_nan_when_eval_loss_is_nan():
    def nan_loss(logits, targets):
        return next_token_loss(logits, targets) * float("nan")

    state, records = run(loss_fn=nan_loss, eval_perplexity=True)

    assert state.global_step == 3
    assert [record["step"] for record in records] == [2, 3]
    assert all(math.isnan(record["metrics"]["eval/perplexity"]) for record in records)
