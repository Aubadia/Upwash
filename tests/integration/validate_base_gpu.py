"""Explicit GPU acceptance test; run only from validate_base.sbatch.

Oracle: Transformers 5.22.2 eager causal attention, Torch 2.15.0, BF16.
FA3 and compiled FA3 use independent model/optimizer copies of the same state.
"""
import json
import os
from pathlib import Path

import torch
from upwash.adaptor import build_llama_model, causal_lm_loss, export_hf_model
from upwash.components.optimizers.muon_adamw import build_muon_adamw
from upwash.config import ModelConfig, OptimizerConfig


def relative_error(actual, expected):
    return float((actual.float()-expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'GPU acceptance must run inside sbatch'
    assert torch.cuda.get_device_capability()[0] == 9, 'FA3 validation requires Hopper'
    torch.manual_seed(2)
    cfg=ModelConfig(vocab_size=128,n_layers=3,d_model=128,d_ff=256,n_heads=2,n_kv_heads=1,
                    max_seq_len=128,rope_theta=500000.0,norm_eps=1e-5,
                    embedding_std=.1,output_std_coeff=1.0,residual_scale_by_depth=True)
    optim_cfg=OptimizerConfig(lr=.004,muon_beta=.9,adam_beta1=.9,adam_beta2=.99,
                              adam_lr_multiplier=1.6,weight_decay=.04)
    initial={k:v.detach().clone() for k,v in build_llama_model(cfg).state_dict().items()}
    x=torch.randint(0,128,(2,128),device='cuda')
    y=torch.randint(0,128,(2,128),device='cuda')
    results={}
    snapshots={}
    for label,implementation,compile_model in [('reference','eager',False),('fa3','flash_attention_3',False),('compiled_fa3','flash_attention_3',True)]:
        model=build_llama_model(cfg).cuda()
        model.load_state_dict(initial)
        model.set_attn_implementation(implementation)
        optimizer=build_muon_adamw(model,config=optim_cfg,model_config=cfg)
        forward=torch.compile(model,backend='inductor',mode='default') if compile_model else model
        for step in range(3):
            optimizer.zero_grad()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=forward(x.clone())
                loss=causal_lm_loss(outputs,y.clone())
            assert torch.isfinite(loss)
            loss.backward()
            if step==0:
                logits=outputs.logits.detach().cpu().float()
                gradients=torch.cat([p.grad.detach().flatten().float().cpu() for p in model.parameters()])
            optimizer.step()
        delta=torch.cat([(p.detach().cpu()-initial[n]).flatten().float() for n,p in model.named_parameters()])
        assert torch.isfinite(delta).all() and delta.norm()>1e-3
        snapshots[label]=(logits,gradients,delta)
        results[label]={'final_loss':float(loss),'update_norm':float(delta.norm())}
        if label!='reference':
            ref=snapshots['reference']
            errors={name:relative_error(a,b) for name,a,b in zip(('logits','gradients','updates'),snapshots[label],ref)}
            assert errors['logits']<.03,errors
            assert errors['gradients']<.03,errors
            assert errors['updates']<.10,errors  # BF16 Newton-Schulz can amplify gradient roundoff.
            results[label]['relative_errors']=errors
        if label=='fa3':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    model(x).logits.sum().backward()
                torch.cuda.synchronize()
            kernels=[e.key for e in prof.key_averages() if 'flash_attn_3' in e.key or 'flash_fwd' in e.key or 'flash_bwd' in e.key]
            assert kernels, 'Profiler did not observe FA3 operators/kernels'
            results[label]['kernels']=kernels
        if label=='compiled_fa3':
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                expected=model(x).logits.float()
                exported=export_hf_model(model,cfg).eval()
                observed=exported(x).logits.float()
            error=relative_error(observed,expected)
            assert error<.03,error
            results[label]['export_relative_error']=error
        del optimizer,forward,model
        torch.cuda.empty_cache()
    destination=Path('/work/asciibase64/upwash/validation')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'gpu-numerics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    print('GPU_NUMERICS_OK',flush=True)


if __name__=='__main__':
    main()
