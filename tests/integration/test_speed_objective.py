"""Experimental composition preserves the actual Trainer's training/eval behavior."""
from copy import deepcopy

import torch

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.components.schedulers.wsd import build_wsd_scheduler
from upwash.config import ModelConfig, ScheduleConfig
from upwash.core.logger import MemoryLogger
from upwash.core.trainer import Trainer


def test_target_aware_objective_matches_native_trainer_updates_schedule_and_evaluation():
    cfg = ModelConfig(vocab_size=32, n_layers=2, d_model=16, d_ff=24, n_heads=2,
                      n_kv_heads=1, max_seq_len=8, rope_theta=500000., norm_eps=1e-5,
                      embedding_std=.1, output_std_coeff=1., residual_scale_by_depth=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        reference = build_llama_model(cfg)
    candidate = deepcopy(reference)
    gen = torch.Generator().manual_seed(6)
    batches = [(torch.randint(32, (2, 8), generator=gen), torch.randint(32, (2, 8), generator=gen))
               for _ in range(4)]
    dev = batches[:2]
    schedule = ScheduleConfig(warmup_ratio=.25, decay_ratio=.5, decay_shape='power', decay_power=1.)
    def kwargs(model, logger):
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        return dict(optimizer=optimizer, lr_scheduler=build_wsd_scheduler(optimizer, config=schedule, total_steps=4),
                    logger=logger, logging_steps=1, eval_steps=2, count_tokens=True,
                    max_grad_norm=1., eval_perplexity=True)
    ref_logger = MemoryLogger()
    ref_trainer = Trainer(model=reference, loss_fn=causal_lm_loss, **kwargs(reference, ref_logger))
    ref_state = ref_trainer.train([(x.clone(), y.clone()) for x,y in batches], eval_batches=dev)
    ref_evaluation = ref_trainer.evaluate(dev)

    # Real reference has completed before the experimental adapter is used.
    from experiments.speed150m.bench import Objective
    from experiments.speed150m.train_fast import ObjectiveTrainer
    actual_logger = MemoryLogger()
    objective = Objective(candidate, cfg)
    actual = ObjectiveTrainer(model=objective, loss_fn=causal_lm_loss, **kwargs(candidate, actual_logger))
    state = actual.train([(x.clone(), y.clone()) for x,y in batches], eval_batches=dev)
    assert state.global_step == ref_state.global_step == 4
    assert actual.evaluate(dev) == ref_evaluation
    assert candidate.lm_head.weight is candidate.model.embed_tokens.weight
    for a,b in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a,b, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(a.grad,b.grad, rtol=1e-6, atol=1e-7)
    for actual_event, expected_event in zip(actual_logger.records, ref_logger.records, strict=True):
        assert actual_event['step'] == expected_event['step']
        for key,value in expected_event['metrics'].items():
            if key != 'train/tokens_per_second':
                assert actual_event['metrics'][key] == value


def test_compiled_update_preserves_muon_adamw_warmup_decay_and_float_lrs(make_optimizer_config):
    from upwash.components.optimizers.muon_adamw import build_muon_adamw
    from experiments.speed150m.compiled_update import compile_update
    cfg = ModelConfig(vocab_size=16, n_layers=1, d_model=8, d_ff=16, n_heads=2,
                      n_kv_heads=1, max_seq_len=4, rope_theta=500000., norm_eps=1e-5,
                      embedding_std=.1, output_std_coeff=1.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18)
        reference = build_llama_model(cfg)
    candidate = deepcopy(reference)
    optimizer_cfg = make_optimizer_config()
    schedule_cfg = ScheduleConfig(warmup_ratio=.25, decay_ratio=.5, decay_shape='power', decay_power=1.)
    batches = [(torch.tensor([[1,2,3,4]]),torch.tensor([[2,3,4,5]])) for _ in range(12)]
    outcomes = []
    for model, compiled in [(reference,False),(candidate,True)]:
        optimizer = build_muon_adamw(model, config=optimizer_cfg, model_config=cfg)
        scheduler = build_wsd_scheduler(optimizer, config=schedule_cfg, total_steps=12)
        # CPU eager backend tests graph capture and live LR changes; Inductor
        # correctness/performance is a separate sbatch GPU acceptance gate.
        update = compile_update(optimizer, plateau_lrs=scheduler.base_lrs, backend='eager') if compiled else optimizer.step
        rates = []
        for x,y in batches:
            rates.append([g['lr'] for g in optimizer.param_groups])
            optimizer.zero_grad()
            causal_lm_loss(model(x.clone()), y.clone()).backward()
            update()
            assert all(isinstance(g['lr'],float) for g in optimizer.param_groups)
            scheduler.step()
        outcomes.append(rates)
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][-1] == [0.,0.]
    for a,b in zip(candidate.parameters(),reference.parameters(),strict=True):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
