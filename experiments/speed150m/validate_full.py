"""Full 150M, actual vocab/context/data numerical acceptance on one allocated GPU."""
import argparse
import json
import os
from pathlib import Path

import torch

from bench import make_forward, optimizer_for, rel
from compiled_update import compile_update
from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.builder import build_dataset
from upwash.config import load_run_config
from upwash.core.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', default='default')
    parser.add_argument('--scope', default='objective')
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--fused-loss', action='store_true')
    parser.add_argument('--eager-fused-loss', action='store_true')
    parser.add_argument('--compile-optimizer', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    cfg = load_run_config('configs/base150m/base150m-30b-seed2-am1.6.json')
    dataset = build_dataset(cfg.train_data)
    iterator = iter(DataLoader(dataset, batch_size=4))
    batches = [tuple(t.cuda() for t in next(iterator)) for _ in range(3)]
    torch.manual_seed(cfg.seed)
    initial_model = build_llama_model(cfg.model)
    initial = {n: p.detach().clone() for n,p in initial_model.named_parameters()}
    del initial_model
    snapshots = {}
    for label in ('reference', 'candidate'):
        torch.manual_seed(cfg.seed)
        model = build_llama_model(cfg.model).cuda()
        model.set_attn_implementation('flash_attention_3')
        optimizer = optimizer_for(model, cfg, args.fused_adam and label == 'candidate')
        update = compile_update(optimizer) if label == 'candidate' and args.compile_optimizer else optimizer.step
        forward = (lambda x,y: causal_lm_loss(model(x),y)) if label == 'reference' else make_forward(model,cfg,args,1)
        losses = []
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
        snapshots[label] = (losses,gradients,deltas)
        del optimizer,update,forward,model
        torch.cuda.empty_cache()
    reference,candidate = snapshots['reference'],snapshots['candidate']
    def aggregate(index):
        # Accumulate norms per parameter to avoid another whole-model concat.
        numerator = sum(float((candidate[index][n]-v).double().square().sum()) for n,v in reference[index].items())
        denominator = sum(float(v.double().square().sum()) for v in reference[index].values())
        return (numerator/max(denominator,1e-24))**.5
    errors = {'loss_relative':max(abs(a-b)/abs(b) for a,b in zip(candidate[0],reference[0])),
              'gradient_relative':aggregate(1),'update_relative':aggregate(2),
              'tied_embedding_gradient_relative':rel(candidate[1]['model.embed_tokens.weight'],reference[1]['model.embed_tokens.weight'])}
    passed = errors['loss_relative'] < .001 and errors['gradient_relative'] < .03 and errors['update_relative'] < .10 and errors['tied_embedding_gradient_relative'] < .03
    result = {'errors':errors, 'passed':passed,'reference_losses':reference[0],'candidate_losses':candidate[0],
              'options':vars(args),'job_id':os.environ['SLURM_JOB_ID'],
              'gradient_errors_by_parameter':{n:rel(candidate[1][n],v) for n,v in reference[1].items()}}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert passed,result


if __name__ == '__main__':
    with torch._dynamo.config.patch(recompile_limit=64, fail_on_recompile_limit_hit=True):
        main()
