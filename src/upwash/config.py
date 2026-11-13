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


@dataclass
class DataConfig:
    root: str
    pattern: str
    format: str
    seq_len: int
    num_tokens: int

    def __post_init__(self):
        if not (self.format == "npy" or self.format == "bin"):
            raise ValueError(f"format must be 'npy' or 'bin', got {self.format!r}")
        if not (self.seq_len > 0):
            raise ValueError(f"seq_len must be greater than 0, got {self.seq_len}")
        if not (self.num_tokens >= self.seq_len):
            raise ValueError(f"num_tokens must be at least seq_len ({self.seq_len}), got {self.num_tokens}")

    def to_dict(self):
        return asdict(self)
    

@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int
    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    embedding_std: float
    output_std_coeff: float

    def __post_init__(self):
        for name in ("vocab_size", "n_layers", "d_model", "d_ff", "n_heads", "n_kv_heads", "max_seq_len"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        
        if not (self.d_model % self.n_heads == 0):
            raise ValueError("d_model must be divied by n_heads")
        if not (self.n_heads % self.n_kv_heads == 0):
            raise ValueError("n_heads must be divied by n_kv_heads")
        
        if not (self.head_dim % 2 == 0):
            raise ValueError
        
        for name in ("rope_theta", "norm_eps", "embedding_std", "output_std_coeff"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0, got {value}")

    def to_dict(self):
        return asdict(self)
    
    def num_parameters(self):
        d = self.d_model
        kv_dim = self.n_kv_heads * self.head_dim

        attention = 2 * d**2 + 2 * d * kv_dim
        mlp = 3 * d * self.d_ff
        norms = 2 * d

        per_layer_= attention + mlp + norms
        return self.vocab_size * d + self.n_layers * per_layer_ + d

    
    @property
    def head_dim(self):
        return self.d_model // self.n_heads

class TrainingConfig:
    def __init__(self, optimizer: OptimizerConfig) -> None:
        self.optimizer = optimizer
