from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from functools import partial
import sys

import torch
import torch._dynamo
import torch.distributed as dist

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_dataset, build_trainer
from upwash.components.kernels.fa3 import register_upwash_fa3_wrapper
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.components.schedulers.wsd import build_wsd_scheduler
from upwash.config import RunConfig
from upwash.core.data import CUDAPrefetchLoader, DataLoader, RankShard
from upwash.core.logger import FlockLogger
from upwash.core.runtime import Runtime, setup_runtime, teardown_runtime
from upwash.core.state import TrainingState
from upwash.core.trainer import Trainer, TrainingProgress


class _CausalLMObject(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, targets):
        output = self.model(inputs)
        loss = causal_lm_loss(output, targets)
        return loss


@dataclass
class TrainResult:
    model: torch.nn.Module
    state: TrainingState


@dataclass(frozen=True)
class RunHooks:
    """Supply a batch stream and optional extensions to the standard LM run."""

    batches: Callable[[Runtime], AbstractContextManager]
    build_model: Callable | None = None
    build_optimizer: Callable | None = None
    prepare: Callable[[Runtime, torch.nn.Module, Trainer], TrainingProgress | None] | None = None
    after_step: Callable[[TrainingProgress], dict | None] | None = None
    schedule_steps: int | None = None


def _train_loader(config, runtime):
    if config.train_data is None:
        raise ValueError("train_data is required without a batch stream")
    dataset = build_dataset(config.train_data)
    if runtime.world_size > 1:
        dataset = RankShard(
            dataset, rank=runtime.rank, world_size=runtime.world_size, batch_size=config.batch_size,
        )

    loader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False)
    if config.prefetch_to_device and torch.device(runtime.device).type == "cuda":
        return CUDAPrefetchLoader(loader, runtime.device)
    return loader


def train(config: RunConfig, *, hooks: RunHooks | None = None) -> TrainResult:
    """Run a causal language model with configured shards or a supplied stream."""
    runtime = setup_runtime(config.device)
    resources = ExitStack()
    completed = False
    try:
        torch.manual_seed(config.seed)

        train_loader = (
            resources.enter_context(hooks.batches(runtime)) if hooks is not None
            else _train_loader(config, runtime)
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

        model_factory = hooks.build_model if hooks and hooks.build_model is not None else build_llama_model
        model = model_factory(config.model).to(runtime.device)

        register_upwash_fa3_wrapper()

        # use "upwash_fa3" to enable upwash fa3 wrapper
        if config.attention_implementation is not None:
            model.set_attn_implementation(config.attention_implementation)

        trained = _CausalLMObject(model) if config.compile_scope == "objective" else model
        compile_enabled = config.compile_scope in ("model", "objective")

        compile_setting = (
            torch._dynamo.config.patch(
                recompile_limit=64,
                fail_on_recompile_limit_hit=True
            )
            if compile_enabled else nullcontext()
        )

        if runtime.world_size > 1:
            trained = torch.nn.parallel.DistributedDataParallel(
                trained,
                device_ids=[torch.device(runtime.device).index] if runtime.device.startswith("cuda") else None,
                forward_sync_buffers=False,
            )
            if compile_enabled:
                trained = torch.compile(trained, mode=config.compile_mode)
        elif compile_enabled:
            trained.compile(mode=config.compile_mode)

        logger = None
        if config.logging is not None and runtime.is_main:
            logger = FlockLogger(project=config.logging.project, name=config.logging.name)
            resources.callback(logger.finish)

        optimizer_factory = hooks.build_optimizer if hooks and hooks.build_optimizer is not None else partial(
            build_muon_adamw, model_config=config.model,
        )
        total_steps = hooks.schedule_steps if hooks and hooks.schedule_steps is not None else len(train_loader)
        trainer = build_trainer(
            config=config.training,
            model=trained,
            loss_fn=None if config.compile_scope == "objective" else causal_lm_loss,
            optimizer_factory=optimizer_factory,
            lr_scheduler_factory=build_wsd_scheduler,
            schedule_config=config.schedule,
            total_steps=total_steps,
            logger=logger,
            count_tokens=True,
            eval_perplexity=True,
            world_size=runtime.world_size,
        )

        train_options = {"eval_batches": eval_loader}
        if hooks is not None:
            if hooks.prepare is not None:
                train_options["progress"] = hooks.prepare(runtime, model, trainer)
            if hooks.after_step is not None:
                train_options["after_step"] = hooks.after_step
        with compile_setting:
            state = trainer.train(train_loader, **train_options)

        if runtime.world_size > 1:
            dist.barrier()
        completed = True
        return TrainResult(model=model, state=state)
    finally:
        try:
            resources.__exit__(*sys.exc_info())
        finally:
            # The launcher will stop peers still inside collectives on a stream failure.
            if completed or hooks is None or runtime.world_size == 1:
                teardown_runtime(runtime)
