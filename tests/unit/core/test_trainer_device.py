import torch
import torch.nn.functional as F

from upwash.core.trainer import Trainer

# A CPU-only machine has no second real device, so the "meta" device stands in
# for an accelerator: a meta model cannot consume CPU tensors, and meta tensors
# carry shapes and devices without data. Trainer must move each batch to the
# device of the model's parameters, as it would for a model placed on CUDA.


def meta_model():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, bias=False)
    return model.to("meta")


def record_input_devices(model):
    devices = []
    model.register_forward_hook(lambda module, inputs, output: devices.append(inputs[0].device.type))
    return devices


def cpu_batches(count=2):
    return [(torch.ones(4, 2), torch.ones(4, 1)) for _ in range(count)]


def test_train_moves_each_batch_to_the_model_device():
    model = meta_model()
    input_devices = record_input_devices(model)
    target_devices = []

    def loss_fn(predictions, targets):
        target_devices.append(targets.device.type)
        return F.mse_loss(predictions, targets)

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    state = trainer.train(cpu_batches(2))

    assert state.global_step == 2
    assert input_devices == ["meta", "meta"]
    assert target_devices == ["meta", "meta"]


def test_evaluate_moves_each_batch_to_the_model_device():
    model = meta_model()
    input_devices = record_input_devices(model)
    target_devices = []

    def loss_fn(predictions, targets):
        target_devices.append(targets.device.type)
        # Meta tensors have no values, so return a real scalar for evaluate's .item().
        return torch.zeros(())

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))

    result = trainer.evaluate(cpu_batches(2))

    assert result["num_targets"] == 8
    assert input_devices == ["meta", "meta"]
    assert target_devices == ["meta", "meta"]
