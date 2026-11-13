import torch
from pathlib import Path

from upwash.core.trainer import Trainer
from upwash.config import TrainingConfig, DataConfig
from upwash.core.data import PackedTokenDataset, Truncate, ConcatDataset

def build_trainer(
        config: TrainingConfig,
        model: torch.nn.Module,
        loss_fn, 
        optimizer_factory,
        lr_scheduler_factory=None,
        schedule_config=None,
        total_steps=None,
        logger=None,
        count_tokens=False,
        pad_token_id=None,

    ) -> Trainer:
    optim = optimizer_factory(model, config=config.optimizer)

    lr_scheduler = (
        lr_scheduler_factory(optim, config=schedule_config, total_steps=total_steps) 
        if lr_scheduler_factory is not None 
        else None
    )

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        logger=logger,
        logging_steps=config.logging_steps,
        count_tokens=count_tokens,
        pad_token_id=pad_token_id,
        eval_steps=config.eval_steps,
        precision=config.precision
    )

def build_dataset(config: DataConfig):
    paths = sorted(Path(config.root).glob(config.pattern))
    if not paths:
        raise ValueError(f"no files match {config.pattern!r} under {config.root!r}")

    if config.format == "npy":
        shards = [PackedTokenDataset.from_npy(path, config.seq_len) for path in paths]
    else:
        shards = [PackedTokenDataset(path, config.seq_len) for path in paths]

    return Truncate(ConcatDataset(shards), num_samples=config.num_tokens // config.seq_len)
