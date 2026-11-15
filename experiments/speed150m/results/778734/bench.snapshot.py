"""Controlled GPU benchmark; invoke through bench.sbatch, never a login GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from upwash.adaptor import build_llama_model, causal_lm_loss, output_multiplier
from upwash.builder import build_dataset
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.config import load_run_config
from upwash.core.data import DataLoader, RankShard, Truncate


class Objective(torch.nn.Module):
    def __init__(self, model, cfg, fused_loss=False, eager_fused_loss=False):
        super().__init__()
        self.model = model
        self.alpha = output_multiplier(cfg)
        self.fused_loss = fused_loss
        if fused_loss:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            self.loss_fn = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)
            if eager_fused_loss:
                self.loss_fn.forward = torch.compiler.disable(self.loss_fn.forward)

    def forward(self, inputs, targets):
        if not self.fused_loss:
            return causal_lm_loss(self.model(inputs), targets)
        hidden = self.model.model(inputs).last_hidden_state
        # Scaling the hidden state preserves the real-valued objective and both
        # tied-weight gradient paths. BF16 rounding differs: numerical gate required.
        hidden = hidden * self.alpha
        return self.loss_fn(self.model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1))


def optimizer_for(model, cfg, fused):
    optimizer = build_muon_adamw(model, config=cfg.training.optimizer, model_config=cfg.model)
    for optim in optimizer.optimizers:
        if isinstance(optim, torch.optim.AdamW) and fused:
            optim.defaults['fused'] = True
            optim.defaults['foreach'] = False
            for group in optim.param_groups:
                group['fused'] = True
                group['foreach'] = False
    return optimizer


def make_forward(model, cfg, args, world):
    objective_scope = args.scope == 'objective' or args.fused_loss
    target = Objective(model, cfg.model, args.fused_loss, getattr(args, 'eager_fused_loss', False)) if objective_scope else model
    if world > 1:
        target = torch.nn.parallel.DistributedDataParallel(
            target, device_ids=[torch.cuda.current_device()], forward_sync_buffers=False,
        )
    if args.mode != 'eager':
        target = torch.compile(target, mode=args.mode)
    if objective_scope:
        return target
    return lambda inputs, targets: causal_lm_loss(target(inputs), targets)


def prefetched(loader, device):
    """One background CPU batch ahead, plus a separate H2D copy stream."""
    iterator = iter(loader)
    copy_stream = torch.cuda.Stream(device=device)
    def fetch():
        batch = next(iterator, None)
        return None if batch is None else tuple(t.pin_memory() for t in batch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fetch)
        def preload():
            nonlocal future
            cpu = future.result()
            if cpu is None:
                return None
            future = pool.submit(fetch)
            with torch.cuda.stream(copy_stream):
                return tuple(t.to(device, non_blocking=True) for t in cpu)
        upcoming = preload()
        while upcoming is not None:
            stream = torch.cuda.current_stream(device)
            stream.wait_stream(copy_stream)
            current = upcoming
            for tensor in current:
                tensor.record_stream(stream)
            upcoming = preload()
            yield current


def max_rank(value, world):
    tensor = torch.tensor(value, device='cuda', dtype=torch.float64)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def benchmark(args, cfg, rank, world):
    local_batch = args.global_batch // world
    assert args.global_batch % world == 0
    total = args.warmup + args.windows * args.steps + (4 if args.profile else 0)
    dataset = Truncate(build_dataset(cfg.train_data), total * args.global_batch)
    dataset = RankShard(dataset, rank=rank, world_size=world, batch_size=local_batch)
    loader = DataLoader(dataset, batch_size=local_batch)
    device = torch.device('cuda', torch.cuda.current_device())
    batches = iter(prefetched(loader, device) if args.prefetch else loader)
    torch.manual_seed(cfg.seed)
    model = build_llama_model(cfg.model).to(device)
    model.set_attn_implementation('flash_attention_3')
    model.train()
    optimizer = optimizer_for(model, cfg, args.fused_adam)
    forward = make_forward(model, cfg, args, world)
    update = torch.compile(optimizer.step) if args.compile_optimizer else optimizer.step
    parameters = list(model.parameters())
    def step():
        torch.compiler.cudagraph_mark_step_begin()
        inputs, targets = next(batches)
        if not args.prefetch:
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = forward(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg.training.max_grad_norm)
        update()
        return loss.detach().clone()
    if world > 1:
        dist.barrier()
    started = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = max_rank(time.perf_counter() - started, world)
    torch.cuda.reset_peak_memory_stats()
    windows = []
    for _ in range(args.windows):
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.steps):
            loss_sum += step()
        torch.cuda.synchronize()
        elapsed = max_rank(time.perf_counter() - started, world)
        if world > 1:
            dist.all_reduce(loss_sum)
        mean_loss = loss_sum.item() / args.steps / world
        assert torch.isfinite(loss_sum).item(), 'nonfinite benchmark loss'
        windows.append({'seconds': elapsed, 'step_ms': elapsed * 1000 / args.steps,
                        'tokens_per_second': args.steps * args.global_batch * cfg.train_data.seq_len / elapsed,
                        'loss': mean_loss})
    memory = {key: max_rank(fn(), world) for key, fn in (
        ('peak_allocated_bytes', torch.cuda.max_memory_allocated),
        ('peak_reserved_bytes', torch.cuda.max_memory_reserved))}
    if args.profile:
        context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                         record_shapes=True) if rank == 0 else nullcontext()
        with context as prof:
            for _ in range(4):
                step()
            torch.cuda.synchronize()
        if rank == 0:
            prof.export_chrome_trace(str(Path(args.output).with_suffix('.trace.json')))
            Path(args.output).with_suffix('.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
    tps = statistics.median(w['tokens_per_second'] for w in windows)
    return {'case': args.name, 'world_size': world, 'local_batch': local_batch,
            'global_batch': args.global_batch, 'windows': windows, **memory,
            'median_tokens_per_second': tps, 'gpu_hours_per_billion_tokens': world * 1e9 / tps / 3600,
            'warmup_and_compile_seconds': warmup_seconds, 'options': vars(args)}


def rel(actual, reference):
    return float((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))


def numerics(args, cfg):
    # Same output vocabulary as the real model, smaller backbone/context for
    # an inexpensive test of the chunked loss and tied embedding gradient.
    cfg.model = dataclasses.replace(cfg.model, n_layers=2, d_model=128, d_ff=256,
                                   n_heads=2, n_kv_heads=1, max_seq_len=128)
    torch.manual_seed(2)
    initial = {k: v.detach().clone() for k, v in build_llama_model(cfg.model).state_dict().items()}
    gen = torch.Generator().manual_seed(17)
    batches = [(torch.randint(cfg.model.vocab_size, (2, 128), generator=gen).cuda(),
                torch.randint(cfg.model.vocab_size, (2, 128), generator=gen).cuda()) for _ in range(3)]
    snapshots = {}
    # Real reference first; retain the existing output multiplier hook.
    for label in ('reference', 'candidate'):
        model = build_llama_model(cfg.model).cuda()
        model.load_state_dict(initial)
        model.set_attn_implementation('flash_attention_3')
        optimizer = optimizer_for(model, cfg, label == 'candidate' and args.fused_adam)
        if label == 'reference':
            forward = lambda x, y: causal_lm_loss(model(x), y)
        else:
            forward = make_forward(model, cfg, args, 1)
        update = torch.compile(optimizer.step) if label == 'candidate' and args.compile_optimizer else optimizer.step
        losses = []
        for index, (x, y) in enumerate(batches):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = forward(x.clone(), y.clone())
            losses.append(loss.item())
            loss.backward()
            if index == 0:
                gradients = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            update()
        deltas = {n: p.detach().cpu() - initial[n] for n, p in model.named_parameters()}
        assert all(torch.isfinite(t).all() for t in [*gradients.values(), *deltas.values()])
        snapshots[label] = (losses, gradients, deltas)
        del update, forward, optimizer, model
        torch.cuda.empty_cache()
    ref, actual = snapshots['reference'], snapshots['candidate']
    errors = {'loss_relative': max(abs(a-b)/abs(b) for a,b in zip(actual[0], ref[0])),
              'gradient_relative': rel(torch.cat([v.flatten() for v in actual[1].values()]),
                                       torch.cat([v.flatten() for v in ref[1].values()])),
              'update_relative': rel(torch.cat([v.flatten() for v in actual[2].values()]),
                                     torch.cat([v.flatten() for v in ref[2].values()])),
              'tied_embedding_gradient_relative': rel(actual[1]['model.embed_tokens.weight'], ref[1]['model.embed_tokens.weight'])}
    result = {'case': args.name, 'errors': errors, 'reference_losses': ref[0], 'candidate_losses': actual[0],
              'passed': errors['loss_relative'] < .001 and errors['gradient_relative'] < .03 and
                        errors['update_relative'] < .10 and errors['tied_embedding_gradient_relative'] < .03}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    assert result['passed'], result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base150m/base150m-30b-seed2-am1.6.json')
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', default='default')
    parser.add_argument('--scope', choices=['model', 'objective'], default='model')
    parser.add_argument('--fused-loss', action='store_true')
    parser.add_argument('--eager-fused-loss', action='store_true')
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--compile-optimizer', action='store_true')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--numerics', action='store_true')
    parser.add_argument('--global-batch', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--windows', type=int, default=3)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires sbatch'
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if world > 1:
        dist.init_process_group('nccl')
    cfg = load_run_config(args.config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with torch._dynamo.config.patch(recompile_limit=64, fail_on_recompile_limit_hit=True):
        result = numerics(args, cfg) if args.numerics else benchmark(args, cfg, rank, world)
    if rank == 0:
        result['versions'] = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'triton']}
        result['job_id'] = os.environ['SLURM_JOB_ID']
        result['node'] = os.uname().nodename
        result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
        print('RESULT', json.dumps(result), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
