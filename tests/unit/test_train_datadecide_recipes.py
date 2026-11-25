import pytest

from scripts.train_datadecide_recipes import native_documents, mix_documents, official_decision


def test_native_documents_rejoin_rows_and_drop_file_edges():
    rows = [(0, [9, 0, 2]), (3, [3, 0, 4, 0, 8])]
    assert list(native_documents(rows, eos=0)) == [(2, [2, 3]), (5, [4])]
    assert list(native_documents(rows[1:], eos=0, resume_offset=5)) == [(5, [4])]


def test_mixture_resume_reproduces_next_document():
    options = [
        [(0, 2, [2, 3]), (0, 5, [4, 5])],
        [(0, 1, [7]), (0, 3, [8])],
    ]

    def open_at(group, pointer):
        if pointer is None:
            return iter(())
        return (doc for doc in options[group] if doc[:2] >= pointer)

    whole = list(mix_documents(open_at, [6, 3], (((0, 0), (0, 0)), (0, 0))))
    assert [doc for _, doc in whole] == [[2, 3], [7], [4, 5], [8]]
    assert list(mix_documents(open_at, [6, 3], whole[2][0])) == whole[2:]


def test_official_threshold_is_inclusive():
    assert official_decision(1.250001, False, None) == (False, False)
    assert official_decision(1.25, False, None) == (True, False)
    assert official_decision(1.25, False, 1.250001) == (True, False)
    assert official_decision(1.25, False, 1.25) == (True, True)
    assert official_decision(1.3, True, 1.24) == (True, False)


def test_flock_summary_accepts_early_checkpoint_with_matching_scores(tmp_path):
    import json
    import os

    from scripts.sync_dclm_flock import final_summary

    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"checkpoint")
    final = tmp_path / "final.pt"
    os.link(latest, final)
    metrics = {"eval/bpb": 1.25, "paloma/macro_bpb": 1.24, "paloma/qualified": True}
    completion = {
        "step": 5000, "tokens": 655360000, "checkpoint": str(final), "metrics": metrics,
        "qualified": True, "stop_reason": "paloma_threshold",
    }
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    config = {
        "recipe": "falcon-and-cc-qc-orig-10p", "planned_steps": 228881,
        "tokens_per_step": 131072, "paloma_threshold": 1.25,
    }
    observed = {"step": 5000, "train/total_tokens_seen": 655360000, **metrics}
    summary = final_summary(tmp_path, config, observed)
    assert summary["paloma_ppl"] == 1.24
    assert summary["oj_total_tokens"] == summary["total_tokens"] == 655360000
    with pytest.raises(RuntimeError, match="disagrees"):
        final_summary(tmp_path, config, {**observed, "paloma/macro_bpb": 1.23})


@pytest.mark.parametrize(
    "step,early_stop,official_expected,continue_training",
    [(5000, False, False, True), (5000, True, True, False), (228881, False, True, True)],
)
def test_checkpoint_gate_controls_batch_iterator(
    monkeypatch, step, early_stop, official_expected, continue_training,
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from scripts import train_datadecide_recipes as runner

    events = []

    @contextmanager
    def upstream(run, runtime):
        def batches():
            events.append("first batch")
            yield 1
            events.append("second batch")
            yield 2

        yield batches()

    def local_validation(run, progress):
        events.append("local")
        return {"eval/bpb": 1.25}

    def official_validation(*args):
        events.append("official")
        return {"eval/bpb": 1.25, "paloma/macro_bpb": 1.25}, {}, True

    monkeypatch.setattr(runner.DCLMRun, "batches", upstream)
    monkeypatch.setattr(runner.DCLMRun, "after_step", local_validation)
    monkeypatch.setattr(runner, "confirmed_metrics", official_validation)
    monkeypatch.setattr(runner.dist, "new_group", lambda **kwargs: object())
    monkeypatch.setattr(runner.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    run = runner.RecipeRun.__new__(runner.RecipeRun)
    run.stop = False
    run.limit = 228881
    run.args = SimpleNamespace(run_dir="unused", early_stop=early_stop)
    run.checkpoint = "saved.pt"
    run.runtime = SimpleNamespace(is_main=True)
    with run.batches(run.runtime) as batches:
        assert next(batches) == 1
        metrics = run.after_step(SimpleNamespace(step=step))
        assert list(batches) == ([2] if continue_training else [])
    assert ("paloma/macro_bpb" in metrics) == official_expected
    assert events == [
        "first batch", "local",
        *(["official"] if official_expected else []),
        *(["second batch"] if continue_training else []),
    ]


@pytest.mark.parametrize("at_horizon", [False, True])
def test_full_horizon_resume_evaluates_official_only_at_endpoint(monkeypatch, tmp_path, at_horizon):
    import json
    from types import SimpleNamespace

    from scripts import train_datadecide_recipes as runner

    step = runner.TOTAL_STEPS if at_horizon else 5000
    tokens = step * runner.TOKENS_PER_STEP
    config = {"planned_steps": runner.TOTAL_STEPS}
    local = {"eval/bpb": 1.2}
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    events = tmp_path / "metrics.jsonl"
    events.write_text(json.dumps({"step": step, "train/total_tokens_seen": tokens, **local}) + "\n")
    monkeypatch.setattr(runner, "RecipeRun", lambda args: SimpleNamespace(config=config))
    monkeypatch.setattr(runner.torch, "load", lambda *args, **kwargs: {
        "config": config, "step": step, "tokens": tokens,
    })

    def confirm(run_dir, path, saved_step, metrics, final):
        assert final and saved_step == runner.TOTAL_STEPS and path == checkpoint
        return (
            {**metrics, "paloma/macro_bpb": 1.19, "paloma/qualified": True},
            {"checkpoint_sha256": "digest"}, True,
        )

    monkeypatch.setattr(runner, "confirmed_metrics", confirm)
    assert runner.recover(SimpleNamespace(run_dir=tmp_path, early_stop=False)) == (75 if at_horizon else 0)
    assert (tmp_path / "completion.json").exists() == at_horizon
    if at_horizon:
        completion = json.loads((tmp_path / "completion.json").read_text())
        assert completion["stop_reason"] == "token_limit"
        assert completion["tokens"] == 29999890432
        assert (tmp_path / "final.pt").samefile(checkpoint)
