from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_dataset, build_trainer
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.components.schedulers.wsd import build_wsd_scheduler
from upwash.config import RunConfig
from upwash.core.data import CUDAPrefetchLoader, DataLoader, RankShard
from upwash.core.state import TrainingState
from upwash.core.logger import FlockLogger
from upwash.core.runtime import setup_runtime, teardown_runtime


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


def train(config: RunConfig) -> TrainResult:
    runtime = setup_runtime(config.device)
    logger = None
    try:
        torch.manual_seed(config.seed)

        train_dataset = build_dataset(config.train_data)
        if runtime.world_size > 1:
            train_dataset = RankShard(
                train_dataset, 
                rank=runtime.rank, 
                world_size=runtime.world_size,
                batch_size=config.batch_size
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            drop_last=False,
        )
        if config.prefetch_to_device and torch.device(runtime.device).type == "cuda":
            train_loader = CUDAPrefetchLoader(train_loader, runtime.device)
        eval_loader = (
            DataLoader(
                build_dataset(config.eval_data),
                batch_size=config.eval_batch_size,
                drop_last=False,
            )
            if config.eval_data is not None
            else None
        )

        model = build_llama_model(config.model).to(runtime.device)
        
        if config.attention_implementation is not None:
            model.set_attn_implementation(config.attention_implementation)

        trained = model
        
        if config.compile_scope == "model":
            model.compile(mode=config.compile_mode)
        elif config.compile_scope == "objective":
            trained = _CausalLMObject(model)
            trained.compile(mode=config.compile_mode)
        
        if runtime.world_size > 1:
            trained = torch.nn.parallel.DistributedDataParallel(
                trained,
                device_ids=[torch.device(runtime.device).index] if runtime.device.startswith("cuda") else None,
                forward_sync_buffers=False,
            )

        if config.logging is not None and runtime.is_main:
            logger = FlockLogger(project=config.logging.project, name=config.logging.name)

        trainer = build_trainer(
            config=config.training,
            model=trained,
            loss_fn=None if config.compile_scope == "objective" else causal_lm_loss,
            optimizer_factory=partial(build_muon_adamw, model_config=config.model),
            lr_scheduler_factory=build_wsd_scheduler,
            schedule_config=config.schedule,
            total_steps=len(train_loader),
            logger=logger,
            count_tokens=True,
            eval_perplexity=True,
            world_size=runtime.world_size,
        )
        state = trainer.train(train_loader, eval_batches=eval_loader)
        if runtime.world_size > 1:
            dist.barrier()
        return TrainResult(model=model, state=state)
    finally:
        if logger is not None:
            # logger can automatically handle this, but add additional protection for safe
            logger.finish()
        teardown_runtime(runtime)
