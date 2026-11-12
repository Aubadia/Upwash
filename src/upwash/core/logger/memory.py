import copy

class MemoryLogger:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, metrics: dict, step):
        record = {
            "step": step,
            "metrics": copy.deepcopy(metrics)   # dict is a mutable object
        }

        self.records.append(record)