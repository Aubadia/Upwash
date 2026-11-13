import torch

from upwash.core.trainer import Trainer
from upwash.config import TrainingConfig

def build_trainer(
        config: TrainingConfig, 
        model: torch.nn.Module, 
        loss_fn, optimizer_factory, 
        logger=None,
        count_tokens=False,
        pad_token_id=None,
    ) -> Trainer:
    optim = optimizer_factory(model.parameters(), **config.optimizer.to_dict())
    return Trainer(
        model=model, 
        loss_fn=loss_fn, 
        optimizer=optim, 
        logger=logger,
        logging_steps=config.logging_steps,
        count_tokens=count_tokens,
        pad_token_id=pad_token_id,
        eval_steps=config.eval_steps,
    )
