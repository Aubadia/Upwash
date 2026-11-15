"""Opt-in manifest adapter; core and ordinary RunConfig semantics are unchanged."""
import argparse
from copy import deepcopy
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from upwash.components.optimizers.combined import CombinedOptimizer
from upwash.config import RunConfig


def build_grouped_optimizer(model, config, specs, *, fused):
    assignments = {spec['name']: [] for spec in specs}
    if len(assignments) != len(specs):
        raise ValueError('optimizer group names must be unique')
    parameters = {spec['name']: [] for spec in specs}
    # named_parameters deduplicates the tied input embedding/readout parameter.
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [s for s in specs if any(fnmatchcase(name, pattern)
                                          for pattern in s['parameter_patterns'])]
        if len(matches) != 1:
            raise ValueError(f'{name}: expected exactly one optimizer group, got {len(matches)}')
        group = matches[0]['name']
        assignments[group].append(name)
        parameters[group].append(parameter)
    muon, adam = [], []
    for spec in specs:
        if not parameters[spec['name']]:
            raise ValueError(f"empty optimizer group: {spec['name']}")
        if not math.isfinite(spec['lr']) or spec['lr'] <= 0:
            raise ValueError('group lr must be finite and positive')
        if not math.isfinite(spec['weight_decay']) or spec['weight_decay'] < 0:
            raise ValueError('group weight_decay must be finite and nonnegative')
        group = dict(params=parameters[spec['name']], name=spec['name'],
                     lr=spec['lr'], weight_decay=spec['weight_decay'])
        if spec['optimizer'] == 'Muon':
            if spec['adjust_lr_fn'] != 'match_rms_adamw':
                raise ValueError('only match_rms_adamw is supported')
            if any(p.ndim != 2 for p in group['params']):
                raise ValueError('Muon requires matrix parameters')
            group['adjust_lr_fn'] = spec['adjust_lr_fn']
            muon.append(group)
        elif spec['optimizer'] == 'AdamW':
            adam.append(group)
        else:
            raise ValueError(f"unsupported optimizer: {spec['optimizer']}")
    optimizers = []
    if muon:
        optimizers.append(torch.optim.Muon(muon, momentum=config.muon_beta,
                                           adjust_lr_fn='match_rms_adamw'))
    if adam:
        optimizers.append(torch.optim.AdamW(adam, betas=(config.adam_beta1, config.adam_beta2),
                                            fused=fused, foreach=False if fused else None))
    return CombinedOptimizer(optimizers), assignments


def read_manifest(path, *, smoke_steps=None, output_dir=None):
    manifest = json.loads(Path(path).read_text())
    raw = deepcopy(manifest['run_config'])
    if smoke_steps is not None:
        if smoke_steps <= 0:
            raise ValueError('smoke_steps must be positive')
        raw['train_data']['num_tokens'] = smoke_steps * manifest['execution']['tokens_per_update']
        raw['training']['eval_steps'] = 100
        raw['training']['logging_steps'] = 20
        raw['schedule']['warmup_ratio'] = .05
        raw['schedule']['decay_ratio'] = .2
        raw['logging'] = None
    if output_dir:
        raw['output_dir'] = output_dir
    return manifest, RunConfig.from_dict(raw)


def run(args):
    from experiments.speed150m import train_fast
    manifest, cfg = read_manifest(args.manifest, smoke_steps=args.smoke_steps,
                                   output_dir=args.output_dir)
    if int(os.environ.get('WORLD_SIZE', '1')) != manifest['execution']['world_size']:
        raise ValueError('manifest requires eight ranks and global batch 32')
    ledger = {}
    original_metadata = train_fast.run_metadata
    def optimizer_for(model, config, fused):
        optimizer, assignment = build_grouped_optimizer(
            model, config.training.optimizer, manifest['required_optimizer_groups'], fused=fused)
        ledger.update(assignment)
        return optimizer
    def metadata(config, runtime, total_steps):
        result = original_metadata(config, runtime, total_steps)
        result['required_optimizer_groups'] = manifest['required_optimizer_groups']
        result['optimizer_group_assignment'] = ledger
        result['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
        result['group_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['base_scale_lr'] = manifest['base_scale_lr']
        result['weight_decay_policy'] = manifest['weight_decay_policy']
        return result
    # Hooks are local to this opt-in executable process, before WSD construction.
    train_fast.load_run_config = lambda _: cfg
    train_fast.optimizer_for = optimizer_for
    train_fast.run_metadata = metadata
    args.config = args.manifest
    train_fast.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--mode', default='default')
    parser.add_argument('--scope', default='objective', choices=['objective', 'model'])
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--compile-optimizer', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.set_defaults(fused_loss=False, eager_fused_loss=False)
    run(parser.parse_args())
