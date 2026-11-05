import torch

class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def compute_loss(
            self, 
            inputs: torch.Tensor, 
            targets: torch.Tensor
    ) -> torch.Tensor:
        out = self.model.forward(inputs)
        return self.loss_fn(out, targets)