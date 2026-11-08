from upwash.core.logger import MemoryLogger


def test_memory_logger_records_multiple_metrics_at_one_step():
    logger = MemoryLogger()
    metrics = {
        "train/loss": 4.0,
        "train/accuracy": 0.75,
        "optimizer/lr": 0.1,
    }

    logger.log(metrics, step=1)

    assert logger.records == [
        {
            "step": 1,
            "metrics": {
                "train/loss": 4.0,
                "train/accuracy": 0.75,
                "optimizer/lr": 0.1,
            },
        }
    ]


def test_memory_logger_preserves_records_in_logging_order():
    logger = MemoryLogger()

    logger.log({"train/loss": 4.0}, step=1)
    logger.log({"train/loss": 2.56}, step=2)

    assert logger.records == [
        {"step": 1, "metrics": {"train/loss": 4.0}},
        {"step": 2, "metrics": {"train/loss": 2.56}},
    ]


def test_memory_logger_instances_have_independent_records():
    first = MemoryLogger()
    second = MemoryLogger()

    first.log({"train/loss": 4.0}, step=1)
    second.log({"train/accuracy": 0.75}, step=3)

    assert first.records == [{"step": 1, "metrics": {"train/loss": 4.0}}]
    assert second.records == [{"step": 3, "metrics": {"train/accuracy": 0.75}}]


def test_memory_logger_preserves_metrics_when_caller_changes_input():
    logger = MemoryLogger()
    metrics = {"train/loss": 4.0}
    logger.log(metrics, step=1)

    metrics["train/loss"] = 2.56
    metrics["train/accuracy"] = 0.75

    assert logger.records == [{"step": 1, "metrics": {"train/loss": 4.0}}]
