"""300M experiment adapter versus independent Torch 2.15.0 optimizer groups."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.config import ModelConfig, OptimizerConfig, ScheduleConfig
from upwash.components.schedulers.wsd import build_wsd_scheduler


@pytest.mark.parametrize('label', ['0.002', '0.0025'])
def test_manifest_groups_match_torch_updates_and_scheduled_weight_decay(label):
    manifest = json.loads((Path(__file__).parents[2] / 'experiments/sweep300m' /
                           f'300m-60b-base-lr{label}-power-p1.json').read_text())
    model_cfg = ModelConfig(vocab_size=16, n_layers=1, d_model=8, d_ff=16,
                           n_heads=2, n_kv_heads=1, max_seq_len=4,
                           rope_theta=500000., norm_eps=1e-5, embedding_std=.1,
                           output_std_coeff=1., residual_scale_by_depth=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18)
        reference = build_llama_model(model_cfg)
    candidate = deepcopy(reference)
    cfg = OptimizerConfig(**manifest['run_config']['training']['optimizer'])
    specs = manifest['required_optimizer_groups']
    named = dict(reference.named_parameters())
    # Independent role assignment; do not reuse the candidate pattern adapter.
    ref_groups = [[], [], []]
    for name, parameter in named.items():
        index = 1 if '.mlp.down_proj.' in name else (
            0 if parameter.ndim == 2 and 'embed_tokens' not in name else 2)
        ref_groups[index].append(parameter)
    muon = torch.optim.Muon([
        dict(params=ref_groups[i], lr=specs[i]['lr'], weight_decay=specs[i]['weight_decay'])
        for i in (0, 1)], momentum=cfg.muon_beta, adjust_lr_fn='match_rms_adamw')
    adam = torch.optim.AdamW(ref_groups[2], lr=specs[2]['lr'],
                             weight_decay=specs[2]['weight_decay'],
                             betas=(cfg.adam_beta1, cfg.adam_beta2))
    schedule = ScheduleConfig(warmup_ratio=.25, decay_ratio=.5,
                              decay_shape='power', decay_power=1.)
    rates = []
    for index in range(12):
        progress = index + 1
        factor = progress / 3 if progress <= 3 else (
            1. if progress <= 6 else (12-progress)/6)
        for group, spec in zip([*muon.param_groups, *adam.param_groups], specs, strict=True):
            group['lr'] = spec['lr'] * factor
        rates.append([group['lr'] for group in [*muon.param_groups, *adam.param_groups]])
        muon.zero_grad(); adam.zero_grad()
        causal_lm_loss(reference(torch.tensor([[1,2,3,4]])), torch.tensor([[2,3,4,5]])).backward()
        muon.step(); adam.step()
    # The real independent reference runs successfully before importing the adapter.
    from experiments.sweep300m.train_grouped import build_grouped_optimizer
    optimizer, assignment = build_grouped_optimizer(candidate, cfg, specs, fused=False)
    assert len({id(p) for g in optimizer.param_groups for p in g['params']}) == len(list(candidate.parameters()))
    assert sum(len(g['params']) for g in optimizer.param_groups) == len(list(candidate.parameters()))
    assert [len(assignment[s['name']]) for s in specs] == [len(g) for g in ref_groups]
    scheduler = build_wsd_scheduler(optimizer, config=schedule, total_steps=12)
    for expected_rates in rates:
        assert [g['lr'] for g in optimizer.param_groups] == pytest.approx(expected_rates, rel=1e-14, abs=0)
        assert [g['weight_decay'] for g in optimizer.param_groups] == [s['weight_decay'] for s in specs]
        optimizer.zero_grad()
        causal_lm_loss(candidate(torch.tensor([[1,2,3,4]])), torch.tensor([[2,3,4,5]])).backward()
        optimizer.step()
        scheduler.step()
    for actual, expected in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_group_adapter_rejects_missing_and_overlapping_assignments():
    from experiments.sweep300m.train_grouped import build_grouped_optimizer
    model = torch.nn.Linear(2, 2)
    spec = dict(name='main', parameter_patterns=['weight'], lr=.01,
                weight_decay=.02, optimizer='Muon', adjust_lr_fn='match_rms_adamw')
    with pytest.raises(ValueError, match='bias'):
        build_grouped_optimizer(model, None, [spec], fused=False)
    with pytest.raises(ValueError, match='weight'):
        build_grouped_optimizer(model, None, [spec, dict(spec, name='duplicate')], fused=False)
