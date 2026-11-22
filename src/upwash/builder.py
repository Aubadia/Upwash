import torch
from pathlib import Path

from upwash.core.trainer import Trainer
from upwash.config import TrainingConfig, DataConfig
from upwash.core.data import PackedTokenDataset, Truncate, ConcatDataset, Permute
from upwash.core.lance_windows import LanceWindowDataset

def build_trainer(
        config: TrainingConfig,
        model: torch.nn.Module,
        loss_fn, 
        optimizer_factory,
        lr_scheduler_factory = None,
        schedule_config = None,
        total_steps = None,
        logger = None,
        count_tokens = False,
        pad_token_id = None,
        eval_perplexity = False,
        world_size = 1,
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
        precision=config.precision,
        eval_perplexity=eval_perplexity,
        world_size=world_size,
        max_grad_norm=config.max_grad_norm
    )

def build_dataset(config: DataConfig):
    if config.format == "lance":
        pool = LanceWindowDataset(config.root, version=config.version)
        if pool.seq_len != config.seq_len:
            raise ValueError(f"config seq_len {config.seq_len} does not match the dataset's seq_len {pool.seq_len}")
    else:
        paths = sorted(Path(config.root).glob(config.pattern))
        if not paths:
            raise ValueError(f"no files match {config.pattern!r} under {config.root!r}")

        if config.format == "npy":
            shards = [PackedTokenDataset.from_npy(path, config.seq_len) for path in paths]
        else:
            shards = [PackedTokenDataset(path, config.seq_len) for path in paths]
        pool = ConcatDataset(shards)

    if config.shuffle_seed is not None:
        pool = Permute(pool, config.shuffle_seed)

    return Truncate(pool, num_samples=config.num_tokens // config.seq_len)
