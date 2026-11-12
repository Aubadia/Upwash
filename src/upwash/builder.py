import torch

from upwash.core.trainer import Trainer
from upwash.config import TrainingConfig

def build_trainer(config: TrainingConfig, model: torch.nn.Module, loss_fn, optimizer_factory, logger=None) -> Trainer:
    optim = optimizer_factory(model.parameters(), lr=config.optimizer.lr)
    return Trainer(model=model, loss_fn=loss_fn, optimizer=optim, logger=logger)