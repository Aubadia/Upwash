from dataclasses import replace

import numpy as np
import torch

from scripts.train_dclm_150m import (
    ADAM_DECAY_CONSTANT,
    MODEL,
    OPTIMIZER,
    SCHEDULE,
    TOKENS_PER_STEP,
    TOTAL_STEPS,
    build_legacy_optimizer,
    build_model,
    build_optimizer,
    learning_rates,
    packed_batches,
)
from upwash.adaptor import build_llama_model
from upwash.components.schedulers.wsd import build_wsd_scheduler


TINY = replace(MODEL, vocab_size=32, n_layers=2, d_model=16, d_ff=32,
               n_heads=2, n_kv_heads=1, max_seq_len=8)


def test_complete_batch_packing_and_mid_document_resume():
    docs = [(0, 2, [0, 1]), (0, 3, list(range(2, 12))), (1, 0, list(range(12, 19)))]
    batches = list(packed_batches(iter(docs), cursor=(0, 0, 0), seq_len=3, batch_size=2))
    first, cursor = batches[0]
    np.testing.assert_array_equal(first, [[0, 1, 2, 3], [3, 4, 5, 6]])
    assert cursor == (0, 3, 4)
    resumed = list(packed_batches(iter(docs[1:]), cursor=cursor, seq_len=3, batch_size=2))
    assert len(resumed) == len(batches) - 1
    for (actual, actual_cursor), (expected, expected_cursor) in zip(resumed, batches[1:]):
        np.testing.assert_array_equal(actual, expected)
        assert actual_cursor == expected_cursor


def test_residual_scales_each_branch_and_survives_checkpoint_reload(tmp_path):
    model = build_model(TINY).eval()
    reference = build_llama_model(TINY).eval()
    reference.load_state_dict(model.state_dict())
    for expected, actual in zip(reference.model.layers, model.model.layers, strict=True):
        # Run a decoder directly with constant projections. Norms and attention
        # inputs can't confound the two individual residual contributions.
        for layer in (expected, actual):
            layer.self_attn.forward = lambda hidden_states, **_: (torch.ones_like(hidden_states), None)
            layer.mlp.forward = lambda hidden_states: torch.full_like(hidden_states, 2)
        hidden = torch.zeros(1, 1, TINY.d_model)
        torch.testing.assert_close(actual(hidden), torch.full_like(hidden, 1.5))
        torch.testing.assert_close(expected(hidden), torch.full_like(hidden, 3))

    trained = build_model(TINY).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(trained.state_dict(), path)
    reloaded = build_model(TINY).eval()
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(trained(ids).logits, reloaded(ids).logits, rtol=0, atol=0)


def test_optimizer_partitions_model_and_matches_effective_update():
    model = build_model(TINY)
    optimizer = build_optimizer(model, config=replace(OPTIMIZER, fused_adamw=False), model_config=TINY)
    grouped = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert sorted(grouped) == sorted(id(param) for param in model.parameters())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    muon = optimizer.optimizers[0]
    assert [group["adjust_lr_fn"] for group in muon.param_groups] == ["spectral_unclamped"]
    assert learning_rates(optimizer) == {
        "train/learning_rate/muon": OPTIMIZER.lr,
        "train/learning_rate/adamw": OPTIMIZER.lr * OPTIMIZER.adam_lr_multiplier,
    }
    assert all(group["weight_decay"] == 0 for group in optimizer.param_groups)

    candidates = [model.model.layers[0].mlp.gate_proj.weight,
                  model.model.layers[0].mlp.down_proj.weight]
    references = [
        torch.nn.Parameter(param.detach() * (1 - OPTIMIZER.weight_decay / TINY.d_model))
        for param in candidates
    ]
    for param, reference in zip(candidates, references, strict=True):
        param.grad = torch.randn_like(param)
        reference.grad = param.grad.clone()
    torch.optim.Muon(references, lr=OPTIMIZER.lr, momentum=OPTIMIZER.muon_beta,
                     adjust_lr_fn="spectral_unclamped", weight_decay=0.0).step()
    optimizer.step()
    for param, reference in zip(candidates, references, strict=True):
        torch.testing.assert_close(param, reference, atol=0, rtol=0)


def test_schedule_boundaries_and_resume_lr():
    model = build_model(TINY)
    optimizer = build_optimizer(model, config=replace(OPTIMIZER, fused_adamw=False), model_config=TINY)
    schedule = build_wsd_scheduler(optimizer, config=SCHEDULE, total_steps=TOTAL_STEPS)
    assert TOTAL_STEPS * TOKENS_PER_STEP == 29_999_890_432
    assert (schedule.warmup_steps, schedule.decay_steps, schedule.decay_start) == (114, 91_552, 137_329)
    for _ in range(113):
        schedule.step()
    peak = learning_rates(optimizer)
    assert abs(peak["train/learning_rate/muon"] - 0.0017) < 1e-12
    schedule.update = schedule.decay_start
    schedule._apply_lr()
    assert learning_rates(optimizer) == peak
    schedule.step()
    decayed = learning_rates(optimizer)
    assert 0 < decayed["train/learning_rate/muon"] < peak["train/learning_rate/muon"]
    resumed_optimizer = build_optimizer(
        build_model(TINY), config=replace(OPTIMIZER, fused_adamw=False), model_config=TINY,
    )
    resumed = build_wsd_scheduler(resumed_optimizer, config=SCHEDULE, total_steps=TOTAL_STEPS)
    resumed.load_state_dict(schedule.state_dict())
    assert learning_rates(resumed_optimizer) == decayed
    resumed.load_state_dict({"update": TOTAL_STEPS})
    assert all(rate == 0 for rate in learning_rates(resumed_optimizer).values())


def test_independent_decay_follows_wsd_factor_and_model_width():
    model = build_model(TINY)
    optimizer = build_optimizer(model, config=replace(OPTIMIZER, fused_adamw=False), model_config=TINY)
    parameters = [model.model.layers[0].mlp.gate_proj.weight, model.model.embed_tokens.weight]
    expected = []
    for group, param, decay_constant in zip(
        optimizer.param_groups, parameters, [OPTIMIZER.weight_decay, ADAM_DECAY_CONSTANT], strict=True,
    ):
        group["lr"] *= 0.5
        param.grad = torch.zeros_like(param)
        expected.append(param.detach() * (1 - 0.5 * decay_constant / TINY.d_model))
    optimizer.step()
    for param, target in zip(parameters, expected, strict=True):
        torch.testing.assert_close(param, target, rtol=0, atol=0)


def test_legacy_factory_preserves_existing_checkpoint_group_order():
    optimizer = build_legacy_optimizer(build_model(TINY), config=replace(OPTIMIZER, fused_adamw=False))
    assert [group["label"] for group in optimizer.param_groups] == [
        "muon_main", "muon_main", "muon_down", "adamw",
    ]
    assert [group["adjust_lr_fn"] for group in optimizer.optimizers[0].param_groups] == ["original"] * 3


def _scatter_worker(rank, store):
    from datetime import timedelta

    import pytest
    import torch.distributed as dist

    from scripts import train_dclm_150m as run
    from upwash.core.runtime import Runtime

    run.WORLD_SIZE = 2
    run.GLOBAL_BATCH = 4
    run.SEQ_LEN = 3
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    source = iter([(np.arange(16, dtype=np.int32).reshape(4, 4), (1, 2, 3))])
    batches = run.DistributedBatches(Runtime(rank, 2, "cpu"), source, trace=True)
    try:
        [(inputs, targets)] = list(batches.take(1))
        local = torch.arange(rank * 8, (rank + 1) * 8).reshape(2, 4)
        torch.testing.assert_close(inputs, local[:, :-1])
        torch.testing.assert_close(targets, local[:, 1:])
        if rank == 0:
            assert batches.cursor == (1, 2, 3)
        with pytest.raises(RuntimeError, match="next complete batch"):
            list(batches.take(1))
    finally:
        batches.close()
        dist.destroy_process_group()


def test_rank_scatter_preserves_distinct_windows_and_propagates_exhaustion(tmp_path):
    torch.multiprocessing.spawn(_scatter_worker, args=((tmp_path / "store").as_uri(),),
                                nprocs=2, join=True)


def test_paloma_validation_selection_and_both_decontamination_splits():
    import pyarrow as pa

    from scripts.train_dclm_150m import paloma_data

    validation = "This validation document has enough words to be included in the decontamination index"
    test = "This test document also contains sufficient words to enter the exact decontamination index"
    train = "This train document has enough words too but belongs to a different split"

    class Dataset:
        def to_batches(self, *, columns, batch_size):
            assert columns == ["_hf_split", "text"]
            return [pa.record_batch([["validation", "test", "train"], [validation, test, train]],
                                    names=columns)]

    index, validation_texts = paloma_data(Dataset())
    assert validation_texts == [validation]
    assert index.is_contaminated(validation)
    assert index.is_contaminated(test)
    assert not index.is_contaminated(train)


def test_validation_runs_once_per_saved_checkpoint(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import train_dclm_150m as run

    events = []
    experiment = run.DCLMRun(SimpleNamespace(
        stop_after=run.TOTAL_STEPS, attention="upwash_fa3", run_dir=tmp_path, resume=False,
    ))
    experiment.runtime = SimpleNamespace(is_main=True, device="cpu")
    experiment.model = object()
    experiment.trainer = object()
    experiment.stream = object()
    experiment.tokenizer = object()
    experiment.validation = ["held-out text"]
    monkeypatch.setattr(run, "save_checkpoint", lambda *args, **kwargs: events.append("save"))
    monkeypatch.setattr(run, "evaluate", lambda *args: events.append("eval") or {"eval/bpb": 1.0})
    monkeypatch.setattr(run.dist, "barrier", lambda: None)

    for step in (4999, 5000, 5001, 10000):
        events.clear()
        result = experiment.after_step(run.TrainingProgress(step, step * run.TOKENS_PER_STEP))
        if step in (5000, 10000):
            assert events == ["save", "eval"]
            assert result == {"eval/bpb": 1.0}
        else:
            assert events == []
            assert result is None
