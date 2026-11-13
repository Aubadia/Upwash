import torch

from upwash.core.trainer import Trainer

# Real gradient computation and SGD updates: bf16 mixed precision must train the
# same model as fp32 up to bfloat16 rounding.

TRUE_WEIGHT = torch.tensor([[1.0, -2.0, 3.0, 0.5]])


def regression_batches(steps):
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(8, 4, generator=generator)
    targets = inputs @ TRUE_WEIGHT.T
    return [(inputs, targets)] * steps


def train(precision, steps=20):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 1)
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        precision=precision,
    )
    initial = trainer.evaluate(regression_batches(1))["loss"]
    trainer.train(regression_batches(steps))
    final = trainer.evaluate(regression_batches(1))["loss"]
    return model, initial, final


def test_bf16_training_matches_fp32_up_to_bfloat16_rounding():
    fp32_model, fp32_initial, fp32_final = train("fp32")
    bf16_model, bf16_initial, bf16_final = train("bf16")

    # Both runs learn: about 15.2 -> 0.29 after 20 updates.
    assert fp32_final < fp32_initial / 10
    assert bf16_final < bf16_initial / 10
    # bfloat16 keeps 8 significand bits (relative step 2**-8 ~= 0.004). Weights of
    # magnitude ~2 drift by ~0.003 after 20 updates on CPU; 0.02 leaves headroom
    # while still catching a broken update (fp32 itself moves weights by >1).
    torch.testing.assert_close(bf16_model.weight, fp32_model.weight, rtol=0, atol=2e-2)
    torch.testing.assert_close(bf16_model.bias, fp32_model.bias, rtol=0, atol=2e-2)
    assert abs(bf16_final - fp32_final) < 2e-2
