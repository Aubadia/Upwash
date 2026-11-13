import pytest
import torch

from upwash.core.trainer import Trainer

# "bf16" is mixed precision: forward and loss run under bfloat16 autocast, while
# parameters, gradients, and optimizer updates stay in float32. "fp32" uses no
# autocast. CPU autocast supports bfloat16, so these run without a GPU.


def linear_model():
    with torch.random.fork_rng(devices=[]):
        return torch.nn.Linear(2, 1, bias=False)


def record_output_dtypes(model):
    dtypes = []
    model.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    return dtypes


def make_trainer(model, **kwargs):
    return Trainer(
        model=model,
        loss_fn=torch.nn.MSELoss(reduction="mean"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        **kwargs,
    )


def batches(count=2):
    return [(torch.ones(4, 2), torch.ones(4, 1)) for _ in range(count)]


PRECISIONS = [("bf16", torch.bfloat16), ("fp32", torch.float32)]


@pytest.mark.parametrize(("precision", "expected"), PRECISIONS, ids=["bf16", "fp32"])
def test_train_runs_the_forward_pass_in_the_configured_precision(precision, expected):
    model = linear_model()
    dtypes = record_output_dtypes(model)

    make_trainer(model, precision=precision).train(batches(2))

    assert dtypes == [expected, expected]


@pytest.mark.parametrize(("precision", "expected"), PRECISIONS, ids=["bf16", "fp32"])
def test_evaluate_runs_the_forward_pass_in_the_configured_precision(precision, expected):
    model = linear_model()
    dtypes = record_output_dtypes(model)

    make_trainer(model, precision=precision).evaluate(batches(2))

    assert dtypes == [expected, expected]


@pytest.fixture
def cpu_autocast_default_float16():
    # torch.autocast without an explicit dtype uses the device's default: bfloat16
    # on CPU but float16 on CUDA. Making the CPU default float16 reproduces the CUDA
    # situation on a CPU-only machine; the global setting is restored afterwards.
    previous = torch.get_autocast_dtype("cpu")
    torch.set_autocast_dtype("cpu", torch.float16)
    try:
        yield
    finally:
        torch.set_autocast_dtype("cpu", previous)


@pytest.mark.parametrize("phase", ["train", "evaluate"])
def test_bf16_means_bfloat16_even_when_the_device_default_is_float16(cpu_autocast_default_float16, phase):
    model = linear_model()
    dtypes = record_output_dtypes(model)
    trainer = make_trainer(model, precision="bf16")

    getattr(trainer, phase)(batches(1))

    # On CUDA a dtype-less autocast would silently run float16, which needs a GradScaler.
    assert dtypes == [torch.bfloat16]


def test_precision_defaults_to_fp32():
    model = linear_model()
    dtypes = record_output_dtypes(model)

    make_trainer(model).train(batches(1))

    assert dtypes == [torch.float32]


def test_bf16_training_keeps_parameters_and_gradients_in_float32():
    model = linear_model()
    initial_weight = model.weight.detach().clone()

    make_trainer(model, precision="bf16").train(batches(2))

    # Mixed precision, not a bf16 model: the master weights stay float32.
    assert model.weight.dtype == torch.float32
    assert model.weight.grad.dtype == torch.float32
    assert not torch.equal(model.weight.detach(), initial_weight)


def test_bf16_autocast_does_not_leak_outside_training():
    trainer = make_trainer(linear_model(), precision="bf16")
    trainer.train(batches(1))

    with torch.random.fork_rng(devices=[]):
        other = torch.nn.Linear(2, 1)

    assert other(torch.ones(1, 2)).dtype == torch.float32


@pytest.mark.parametrize("precision", ["fp16", "BF16", "bfloat16", "float32", ""])
def test_trainer_rejects_an_unknown_precision(precision):
    with pytest.raises(ValueError):
        make_trainer(linear_model(), precision=precision)
