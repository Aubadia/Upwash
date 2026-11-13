import torch
from pathlib import Path

from upwash.core.trainer import Trainer
from upwash.config import TrainingConfig, DataConfig
from upwash.core.data import PackedTokenDataset, Truncate, ConcatDataset

def build_trainer(
        config: TrainingConfig,
        model: torch.nn.Module,
        loss_fn, optimizer_factory,
        logger=None,
        count_tokens=False,
        pad_token_id=None,

    ) -> Trainer:
    optim = optimizer_factory(model.parameters(), lr=config.optimizer.lr)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optim,
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
