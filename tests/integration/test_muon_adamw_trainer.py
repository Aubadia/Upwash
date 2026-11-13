from copy import deepcopy
from functools import partial

import torch

from upwash.builder import build_trainer
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.config import ModelConfig, TrainingConfig


class TinyTiedLM(torch.nn.Module):
    """Tensor-output model compatible with Trainer's current loss interface."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.hidden = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, tokens):
        return self.lm_head(torch.tanh(self.hidden(self.embed_tokens(tokens))))


def test_builder_wires_muon_adamw_and_matches_two_reference_updates(make_optimizer_config):
    model_config = ModelConfig(
        vocab_size=8,
        n_layers=1,
        d_model=4,
        d_ff=8,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=8,
        rope_theta=500_000.0,
        norm_eps=1e-5,
        embedding_std=0.1,
        output_std_coeff=1.0,
    )
    config = make_optimizer_config(lr=0.008)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TinyTiedLM().to(device="cpu", dtype=torch.float32)
    reference = deepcopy(model)
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss_fn = torch.nn.CrossEntropyLoss()
    batches = [
        (torch.tensor([1, 2]), torch.tensor([2, 3])),
        (torch.tensor([3, 4]), torch.tensor([4, 5])),
    ]

    # torch 2.15.0+cu132 reference: explicit groups, independent parameters and
    # inputs, and real gradients. Run the oracle before building the Trainer.
    reference_muon = torch.optim.Muon(
        [reference.hidden.weight],
        lr=config.lr,
        momentum=config.muon_beta,
        weight_decay=config.weight_decay / model_config.d_model,
        adjust_lr_fn="match_rms_adamw",
    )
    reference_adam = torch.optim.AdamW(
        [reference.embed_tokens.weight, reference.hidden.bias],
        lr=config.lr * config.adam_lr_multiplier,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay / model_config.d_model,
    )
    reference.train()
    for inputs, targets in batches:
        reference_muon.zero_grad()
        reference_adam.zero_grad()
        loss_fn(reference(inputs.clone()), targets.clone()).backward()
        reference_muon.step()
        reference_adam.step()

    trainer = build_trainer(
        config=TrainingConfig(optimizer=config),
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=partial(build_muon_adamw, model_config=model_config),
    )
    state = trainer.train([(x.clone(), y.clone()) for x, y in batches])

    assert trainer.model is model
    assert state.global_step == 2
    assert model.lm_head.weight is model.embed_tokens.weight
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected = reference_parameters[name]
        assert not torch.equal(parameter.detach(), initial_parameters[name]), name
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-7)
        assert parameter.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-5, atol=1e-7)
