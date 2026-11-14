"""Training entry point.

Single process:  python -m upwash.train --config configs/run.json
One node, 8 GPUs: torchrun --standalone --nproc_per_node 8 -m upwash.train --config configs/run.json
"""

import argparse
import os
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from upwash.adaptor import build_llama_model, causal_lm_loss, export_hf_model, output_multiplier
from upwash.builder import build_dataset, build_trainer
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.components.schedulers.wsd import build_wsd_scheduler
from upwash.config import RunConfig, load_run_config
from upwash.core.data import DataLoader, RankShard
from upwash.core.state import TrainingState
from upwash.core.logger import FlockLogger


@dataclass
class TrainResult:
    model: torch.nn.Module
    state: TrainingState


@dataclass
class _Runtime:
    rank: int = 0
    world_size: int = 1
    device: str = "cpu"
    initialized_here: bool = False

    @property
    def is_main(self):
        return self.rank == 0


def _setup_runtime(config: RunConfig) -> _Runtime:
    """Join the torchrun process group when launched with WORLD_SIZE > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return _Runtime(device=config.device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = config.device.startswith("cuda")
    initialized_here = False
    if not dist.is_initialized():
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        initialized_here = True
    device = f"cuda:{local_rank}" if use_cuda else "cpu"
    return _Runtime(rank=rank, world_size=world_size, device=device, initialized_here=initialized_here)


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(config: RunConfig, runtime: _Runtime, total_steps: int) -> dict:
    """The wandb config: what the lab requires, plus provenance.

    Required: model configuration, initialization, optimizer settings, batch size,
    schedule, and training duration. Provenance: seed, data source, commit.
    Runtime plumbing (device, paths of outputs, logging names) is left out.
    """
    global_batch = config.batch_size * runtime.world_size
    seq_len = config.train_data.seq_len
    return {
        "model": {**config.model.to_dict(), "num_parameters": config.model.num_parameters()},
        "initialization": {
            "embedding_std": config.model.embedding_std,
            "hidden_linear_std": "in_features ** -0.5",
            "output_multiplier": output_multiplier(config.model),
        },
        "optimizer": {
            **config.training.optimizer.to_dict(),
            "max_grad_norm": config.training.max_grad_norm,
            "precision": config.training.precision,
        },
        "batch": {
            "global_batch_size": global_batch,
            "global_batch_tokens": global_batch * seq_len,
            "world_size": runtime.world_size,
        },
        "schedule": config.schedule.to_dict(),
        "duration": {
            "total_steps": total_steps,
            # The budget itself: a smaller final batch makes steps x batch overcount.
            "train_tokens": config.train_data.num_tokens,
            "seq_len": seq_len,
            "eval_steps": config.training.eval_steps,
        },
        "data": {
            "train_root": config.train_data.root,
            "shuffle_seed": config.train_data.shuffle_seed,
            "eval_root": None if config.eval_data is None else config.eval_data.root,
        },
        "seed": config.seed,
        "git_commit": _git_commit(),
    }


def train(config: RunConfig) -> TrainResult:
    runtime = _setup_runtime(config)
    torch.manual_seed(config.seed)

    train_dataset = build_dataset(config.train_data)
    if runtime.world_size > 1:
        train_dataset = RankShard(
            train_dataset, rank=runtime.rank, world_size=runtime.world_size,
            batch_size=config.batch_size,
        )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=False)
    eval_loader = (
        DataLoader(
            build_dataset(config.eval_data),
            batch_size=config.eval_batch_size,
            drop_last=False,
        )
        if config.eval_data is not None
        else None
    )

    # The same seed on every rank gives identical initial weights; DDP also
    # broadcasts rank 0's weights when wrapping.
    model = build_llama_model(config.model).to(runtime.device)
    trained = model
    if runtime.world_size > 1:
        # Rotary buffers are constants (synced once at init), and evaluation runs
        # forward passes outside the training collectives, so buffers are not
        # synced per forward.
        trained = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.device(runtime.device).index] if runtime.device.startswith("cuda") else None,
            forward_sync_buffers=False,
        )

    # Load the tokenizer before training so a missing offline cache fails now,
    # not after the run when the export is written.
    tokenizer = None
    if config.output_dir is not None and config.tokenizer is not None and runtime.is_main:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, revision=config.tokenizer_revision)

    total_steps = len(train_loader)
    logger = None
    if config.logging is not None and runtime.is_main:
        if config.logging.group is not None:
            os.environ.setdefault("WANDB_RUN_GROUP", config.logging.group)
        logger = FlockLogger(
            project=config.logging.project,
            name=config.logging.name,
            config=run_metadata(config, runtime, total_steps),
        )

    try:
        trainer = build_trainer(
            config=config.training,
            model=trained,
            loss_fn=causal_lm_loss,
            optimizer_factory=partial(build_muon_adamw, model_config=config.model),
            lr_scheduler_factory=build_wsd_scheduler,
            schedule_config=config.schedule,
            total_steps=total_steps,
            logger=logger,
            count_tokens=True,
            eval_perplexity=True,
            world_size=runtime.world_size,
        )
        state = trainer.train(train_loader, eval_batches=eval_loader)
        if config.output_dir is not None and runtime.is_main:
            save_model(model, config, tokenizer=tokenizer)
        if runtime.world_size > 1:
            dist.barrier()
        return TrainResult(model=model, state=state)
    finally:
        if logger is not None:
            # logger can automatically handle this, but add additional protection for safe
            logger.finish()
        if runtime.initialized_here:
            dist.destroy_process_group()


def save_model(model, config: RunConfig, *, tokenizer=None) -> Path:
    """Write the alpha-folded HF model (and tokenizer, when given) to output_dir."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = export_hf_model(
        model, config.model, max_position_embeddings=config.export_max_position_embeddings,
    )
    exported.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    return output_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train an Upwash language model.")
    parser.add_argument("--config", required=True, help="Path to a JSON RunConfig.")
    parser.add_argument("--name", help="Override logging.name (for example the Slurm run name).")
    parser.add_argument("--output-dir", help="Override output_dir.")
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    if args.name is not None and config.logging is not None:
        config.logging.name = args.name
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    train(config)


if __name__ == "__main__":
    main()
