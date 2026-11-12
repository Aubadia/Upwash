import math
from dataclasses import dataclass, asdict

@dataclass
class OptimizerConfig:
    lr: float
    muon_beta: float
    adam_beta1: float
    adam_beta2: float
    adam_lr_multiplier: float
    weight_decay: float

    def __post_init__(self):
        if not (math.isfinite(self.lr) and self.lr >= 0):
            raise ValueError(f"lr must be a finite number >= 0, got {self.lr}")
        if not (self.muon_beta < 1 and self.muon_beta >= 0):
            raise ValueError(f"muon_beta must be in [0, 1), got {self.muon_beta}")
        if not (self.adam_beta1 < 1 and self.adam_beta1 >= 0):
            raise ValueError(f"adam_beta1 must be in [0, 1), got {self.adam_beta1}")
        if not (self.adam_beta2 < 1 and self.adam_beta2 >= 0):
            raise ValueError(f"adam_beta2 must be in [0, 1), got {self.adam_beta2}")
        if not (math.isfinite(self.adam_lr_multiplier) and self.adam_lr_multiplier > 0):
            raise ValueError(f"adam_lr_multiplier must be a finite number > 0, got {self.adam_lr_multiplier}")
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError(f"weight_decay must be a finite number >= 0, got {self.weight_decay}")


    def to_dict(self):
        return asdict(self)

@dataclass
class ScheduleConfig: 
    warmup_ratio: float
    decay_ratio: float
    decay_shape: str
    decay_power: float

    def __post_init__(self):
        if not (self.decay_shape == "cosine" or self.decay_shape == "power"):
            raise ValueError(f"decay_shape must be cosine or power, got {self.decay_shape}")
        if not (math.isfinite(self.decay_power) and self.decay_power > 0):
            raise ValueError(f"decay_power must be a finite number > 0, got {self.decay_power}")
        if not (0 <= self.warmup_ratio <= 1):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if not (0 <= self.decay_ratio <= 1):
            raise ValueError(f"decay_ratio must be in [0, 1], got {self.decay_ratio}")
        if self.warmup_ratio + self.decay_ratio > 1:
            raise ValueError(
            f"warmup_ratio + decay_ratio must be at most 1, "
            f"got {self.warmup_ratio} + {self.decay_ratio}"
            )

    def to_dict(self):
        return asdict(self)


class TrainingConfig:
    def __init__(self, optimizer: OptimizerConfig) -> None:
        self.optimizer = optimizer
