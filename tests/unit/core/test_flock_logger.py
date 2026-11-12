import math
from unittest.mock import Mock


class RecordingRun:
    def __init__(self):
        self.calls = []
        self.finish_calls = 0

    def log(self, data, *, step, commit):
        # Keep the received mapping so the test can observe input isolation.
        self.calls.append((data, step, commit))

    def finish(self):
        self.finish_calls += 1


def test_flock_logger_exit_callback_finishes_the_recording_run(monkeypatch):
    from upwash.core.logger import flock

    run = RecordingRun()
    monkeypatch.setattr(flock.wandb, "init", Mock(return_value=run))
    register = Mock()
    monkeypatch.setattr(flock.atexit, "register", register)
    flock.FlockLogger(
        project="lab6-predictable-scaling", name="tiny-training", config={"lr": 0.1}
    )

    # Invoke only this test's registered callback, not the process-wide exit hooks.
    callback = register.call_args.args[0]
    callback()

    assert run.finish_calls == 1


def test_flock_logger_initializes_run_with_project_name_and_config(monkeypatch):
    from upwash.core.logger import flock

    monkeypatch.setattr(flock.atexit, "register", Mock())
    init = Mock(return_value=RecordingRun())
    monkeypatch.setattr(flock.wandb, "init", init)
    config = {"lr": 0.1, "batch_size": 4}

    flock.FlockLogger(
        project="lab6-predictable-scaling", name="tiny-training", config=config
    )

    init.assert_called_once_with(
        project="lab6-predictable-scaling", name="tiny-training", config=config
    )


def test_flock_logger_forwards_one_complete_metric_event_with_an_independent_mapping(monkeypatch):
    from upwash.core.logger import flock

    monkeypatch.setattr(flock.atexit, "register", Mock())
    run = RecordingRun()
    monkeypatch.setattr(flock.wandb, "init", Mock(return_value=run))
    from upwash.core.logger import FlockLogger

    logger = FlockLogger(
        project="lab6-predictable-scaling", name="tiny-training", config={"lr": 0.1}
    )
    # Fixed lab metric payload; aggregation and measurement belong to training code.
    metrics = {
        "train/loss": 4.0,
        "train/grad_norm": 1.5,
        "train/learning_rate": 0.1,
        "train/tokens_per_second": 2048.0,
        "train/total_tokens_seen": 8192,
        "eval/loss": 1.0,
        "eval/perplexity": math.e,
    }
    expected_metrics = metrics.copy()

    logger.log(metrics, step=7)

    assert run.calls == [(expected_metrics, 7, True)]
    assert metrics == expected_metrics

    metrics.clear()

    assert run.calls[0][0] == expected_metrics
