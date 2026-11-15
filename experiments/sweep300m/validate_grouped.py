"""Full-size 300M BF16 numerical gate on real data, only inside sbatch.

Oracle: installed Torch 2.15.0 Muon/AdamW and native HF causal-LM objective.
Uses the same tolerances as the accepted 150M speed recipe.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.speed150m.bench import make_forward, rel
from experiments.speed150m.compiled_update import compile_update
from experiments.sweep300m.train_grouped import build_grouped_optimizer, read_manifest
from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_dataset
from upwash.components.optimizers.combined import CombinedOptimizer
from upwash.core.data import DataLoader


def reference_optimizer(model, config, specs):
    parameters = [[], [], []]
    for name, parameter in model.named_parameters():
        role = 1 if '.mlp.down_proj.' in name else (
            0 if parameter.ndim == 2 and 'embed_tokens' not in name else 2)
        parameters[role].append(parameter)
    return CombinedOptimizer([
        torch.optim.Muon([dict(params=parameters[i], lr=specs[i]['lr'],
                               weight_decay=specs[i]['weight_decay']) for i in (0, 1)],
                         momentum=config.muon_beta, adjust_lr_fn='match_rms_adamw'),
        torch.optim.AdamW(parameters[2], lr=specs[2]['lr'], weight_decay=specs[2]['weight_decay'],
                          betas=(config.adam_beta1, config.adam_beta2))])


def main(args):
    assert os.environ.get('SLURM_JOB_ID'), 'GPU validation requires sbatch'
    assert torch.__version__.split('+')[0] == '2.15.0', 'Re-review oracle on version changes'
    manifest, cfg = read_manifest(args.manifest)
    iterator = iter(DataLoader(build_dataset(cfg.train_data), batch_size=cfg.batch_size))
    batches = [tuple(t.cuda() for t in next(iterator)) for _ in range(3)]
    torch.manual_seed(cfg.seed)
    model = build_llama_model(cfg.model)
    initial = {n:p.detach().clone() for n,p in model.named_parameters()}
    assert sum(p.numel() for p in model.parameters()) == 298273792
    del model
    snapshots = {}
    for label in ('reference', 'candidate'):
        torch.manual_seed(cfg.seed)
        model = build_llama_model(cfg.model).cuda()
        model.set_attn_implementation('flash_attention_3')
        specs = manifest['required_optimizer_groups']
        if label == 'reference':
            optimizer = reference_optimizer(model, cfg.training.optimizer, specs)
            update = optimizer.step
            forward = lambda x,y: causal_lm_loss(model(x),y)
        else:
            optimizer, _ = build_grouped_optimizer(model, cfg.training.optimizer, specs, fused=True)
            update = compile_update(optimizer)
            forward = make_forward(model, cfg, SimpleNamespace(scope='objective', mode='default',
                                                               fused_loss=False), 1)
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for index,(x,y) in enumerate(batches):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = forward(x.clone(),y.clone())
            losses.append(loss.item())
            loss.backward()
            if index == 0:
                gradients = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.max_grad_norm)
            update()
        deltas = {n:p.detach().cpu()-initial[n] for n,p in model.named_parameters()}
        assert all(torch.isfinite(v).all() for v in [*gradients.values(),*deltas.values()])
        snapshots[label] = (losses,gradients,deltas,torch.cuda.max_memory_allocated())
        del optimizer,update,forward,model
        torch.cuda.empty_cache()
    reference,candidate = snapshots['reference'],snapshots['candidate']
    def aggregate(index):
        numerator = sum(float((candidate[index][n]-v).double().square().sum()) for n,v in reference[index].items())
        denominator = sum(float(v.double().square().sum()) for v in reference[index].values())
        return (numerator/max(denominator,1e-24))**.5
    errors = dict(loss_relative=max(abs(a-b)/abs(b) for a,b in zip(candidate[0],reference[0])),
                  gradient_relative=aggregate(1), update_relative=aggregate(2),
                  tied_embedding_gradient_relative=rel(candidate[1]['model.embed_tokens.weight'],
                                                       reference[1]['model.embed_tokens.weight']))
    passed = (errors['loss_relative'] < .001 and errors['gradient_relative'] < .03
              and errors['update_relative'] < .10 and errors['tied_embedding_gradient_relative'] < .03)
    result = dict(passed=passed, errors=errors, reference_losses=reference[0], candidate_losses=candidate[0],
                  peak_allocated_bytes=dict(reference=reference[3],candidate=candidate[3]),
                  torch_version=torch.__version__, job_id=os.environ['SLURM_JOB_ID'])
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert passed,result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    with torch._dynamo.config.patch(recompile_limit=64, fail_on_recompile_limit_hit=True):
        main(parser.parse_args())
