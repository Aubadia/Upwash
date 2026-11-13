import math

from upwash.config import ScheduleConfig


class WSDScheduler:
    def __init__(self, optimizer, *, config: ScheduleConfig, total_steps: int):
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if config.decay_shape not in ("linear", "cosine", "power"):
            raise ValueError("decay_shape must be linear, cosine, or power")

        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * config.warmup_ratio)
        self.decay_steps = int(total_steps * config.decay_ratio)
        self.decay_start = total_steps - self.decay_steps
        self.decay_shape = config.decay_shape
        self.decay_power = config.decay_power
        self.update = 1
        self._apply_lr()

    def step(self):
        self.update += 1
        self._apply_lr()

    def _apply_lr(self):
        if self.update <= self.warmup_steps:
            factor = self.update / self.warmup_steps
        elif self.decay_steps > 0 and self.update > self.decay_start:
            progress = min(1.0, (self.update - self.decay_start) / self.decay_steps)
            if self.decay_shape == "cosine":
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif self.decay_shape == "power":
                factor = (1.0 - progress) ** self.decay_power
            else:
                factor = 1.0 - progress
        else:
            factor = 1.0

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor


def build_wsd_scheduler(optimizer, *, config: ScheduleConfig, total_steps: int) -> WSDScheduler:
    return WSDScheduler(optimizer, config=config, total_steps=total_steps)
