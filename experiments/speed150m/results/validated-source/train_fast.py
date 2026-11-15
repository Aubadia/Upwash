"""Opt-in experimental composition using the real Trainer, WSD and HF export.

Flags are explicit: benchmark results determine which are enabled. No default
training config or source implementation is overwritten by this entry point.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

if __package__:
    from .bench import Objective, optimizer_for, prefetched
    from .compiled_update import compile_update
else:
    from bench import Objective, optimizer_for, prefetched
    from compiled_update import compile_update
from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_dataset
from upwash.components.schedulers.wsd import build_wsd_scheduler
from upwash.config import load_run_config
from upwash.core.data import DataLoader, RankShard
from upwash.core.logger import MemoryLogger, FlockLogger
from upwash.core.trainer import Trainer
from upwash.train import _setup_runtime, run_metadata, save_model


class ObjectiveTrainer(Trainer):
    def compute_loss(self, inputs, targets):
        torch.compiler.cudagraph_mark_step_begin()
        return self.model(inputs, targets)


class DeviceBatches:
    """Reiterable prefetch wrapper: evaluation must not retain an exhausted iterator."""
    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return prefetched(self.loader, self.device)


def run(args):
    assert os.environ.get('SLURM_JOB_ID'), 'GPU execution requires sbatch'
    cfg = load_run_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    runtime = _setup_runtime(cfg)
    torch.manual_seed(cfg.seed)
    data = build_dataset(cfg.train_data)
    if runtime.world_size > 1:
        data = RankShard(data, rank=runtime.rank, world_size=runtime.world_size, batch_size=cfg.batch_size)
    loader = DataLoader(data, batch_size=cfg.batch_size)
    dev = DataLoader(build_dataset(cfg.eval_data), batch_size=cfg.eval_batch_size) if cfg.eval_data else None
    if args.prefetch:
        loader = DeviceBatches(loader, runtime.device)
        if dev is not None:
            dev = DeviceBatches(dev, runtime.device)
    model = build_llama_model(cfg.model).to(runtime.device)
    model.set_attn_implementation(cfg.attention_implementation or 'flash_attention_3')
    objective_scope = args.scope == 'objective' or args.fused_loss
    trained = Objective(model, cfg.model, args.fused_loss, args.eager_fused_loss) if objective_scope else model
    if runtime.world_size > 1:
        trained = torch.nn.parallel.DistributedDataParallel(
            trained, device_ids=[torch.device(runtime.device).index], forward_sync_buffers=False)
    if args.mode != 'eager':
        trained = torch.compile(trained, mode=args.mode)
    optimizer = optimizer_for(model, cfg, args.fused_adam)
    scheduler = build_wsd_scheduler(optimizer, config=cfg.schedule, total_steps=len(loader))
    if args.compile_optimizer:
        optimizer.step = compile_update(optimizer, plateau_lrs=scheduler.base_lrs)
    metadata = run_metadata(cfg, runtime, len(loader))
    experimental_sources = sorted(Path(__file__).parent.glob('*.py'))
    metadata['experimental_sources_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in experimental_sources}
    metadata['speed_optimizations'] = {'mode': args.mode, 'scope': args.scope,
                                     'fused_loss': args.fused_loss,
                                     'eager_fused_loss': args.eager_fused_loss,
                                     'fused_adam': args.fused_adam, 'prefetch': args.prefetch,
                                     'compile_optimizer': args.compile_optimizer}
    logger = None
    if runtime.is_main:
        logger = (FlockLogger(project=cfg.logging.project, name=cfg.logging.name, config=metadata)
                  if args.online and cfg.logging else MemoryLogger())
    trainer_class = ObjectiveTrainer if objective_scope else Trainer
    trainer = trainer_class(model=trained, loss_fn=causal_lm_loss, optimizer=optimizer,
                            lr_scheduler=scheduler, logger=logger,
                            logging_steps=cfg.training.logging_steps,
                            eval_steps=cfg.training.eval_steps, precision=cfg.training.precision,
                            max_grad_norm=cfg.training.max_grad_norm, count_tokens=True,
                            eval_perplexity=True, world_size=runtime.world_size)
    tokenizer = None
    if cfg.output_dir and cfg.tokenizer and runtime.is_main:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, revision=cfg.tokenizer_revision)
    try:
        if runtime.world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch._dynamo.config.patch(recompile_limit=64, fail_on_recompile_limit_hit=True):
            state = trainer.train(loader, eval_batches=dev)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter()-started, device=runtime.device, dtype=torch.float64)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device=runtime.device, dtype=torch.float64)
        if runtime.world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        result = {'steps': state.global_step, 'training_and_eval_seconds': elapsed.item(),
                  'tokens': cfg.train_data.num_tokens,
                  'end_to_end_tokens_per_second': cfg.train_data.num_tokens/elapsed.item(),
                  'peak_allocated_bytes': peak.item(), 'metadata': metadata,
                  'job_id': os.environ['SLURM_JOB_ID']}
        if runtime.is_main:
            if isinstance(logger, MemoryLogger):
                result['history'] = logger.records
            if cfg.output_dir:
                path = save_model(model, cfg, tokenizer=tokenizer)
                # Verify the actual serialized artifact, including output and
                # residual multipliers, against the original model's logits.
                from transformers import LlamaForCausalLM
                reloaded = LlamaForCausalLM.from_pretrained(path).to(runtime.device).eval()
                model.eval()
                x, _ = next(iter(dev if dev is not None else loader))
                x = x[:1, :128].to(runtime.device)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    expected = model(x).logits.float()
                    actual = reloaded(x).logits.float()
                error = float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
                assert error < .03, f'export relative error {error}'
                result['export_relative_error'] = error
            Path(args.result).parent.mkdir(parents=True, exist_ok=True)
            Path(args.result).write_text(json.dumps(result, indent=2)+'\n')
            print('INTEGRATION_RESULT', json.dumps({k:v for k,v in result.items() if k != 'history'}), flush=True)
        if runtime.world_size > 1:
            dist.barrier()
    finally:
        if isinstance(logger, FlockLogger):
            logger.finish()
        if runtime.initialized_here:
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--mode', default='default')
    parser.add_argument('--scope', choices=['model', 'objective'], default='model')
    parser.add_argument('--fused-loss', action='store_true')
    parser.add_argument('--eager-fused-loss', action='store_true')
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--compile-optimizer', action='store_true')
    parser.add_argument('--online', action='store_true')
    run(parser.parse_args())
