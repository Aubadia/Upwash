from dataclasses import dataclass
from functools import partial

import torch

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_dataset, build_trainer
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.components.schedulers.wsd import build_wsd_scheduler
from upwash.config import RunConfig
from upwash.core.data import DataLoader
from upwash.core.state import TrainingState
from upwash.core.logger import FlockLogger


@dataclass
class TrainResult:
    model: torch.nn.Module
    state: TrainingState


def train(config: RunConfig) -> TrainResult:
    torch.manual_seed(config.seed)

    train_loader = DataLoader(
        build_dataset(config.train_data),
        batch_size=config.batch_size,
        drop_last=False,
    )
    eval_loader = (
        DataLoader(
            build_dataset(config.eval_data),
            batch_size=config.eval_batch_size,
            drop_last=False,
        )
        if config.eval_data is not None
        else None
    )

    model = build_llama_model(config.model).to(config.device)
    logger = None
    if config.logging is not None:
        logger = FlockLogger(project=config.logging.project, name=config.logging.name)

    try:
        trainer = build_trainer(
            config=config.training,
            model=model,
            loss_fn=causal_lm_loss,
            optimizer_factory=partial(build_muon_adamw, model_config=config.model),
            lr_scheduler_factory=build_wsd_scheduler,
            schedule_config=config.schedule,
            total_steps=len(train_loader),
            logger=logger,
            count_tokens=True,
            eval_perplexity=True,
        )
        state = trainer.train(train_loader, eval_batches=eval_loader)
        return TrainResult(model=model, state=state)
    finally:
        if logger is not None:
            # logger can automatically handle this, but add additional protection for safe
            logger.finish()
