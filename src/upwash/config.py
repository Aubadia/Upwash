import math

class TrainingConfig:
    def __init__(self, optimizer: OptimizerConfig) -> None:
        self.optimizer = optimizer

class OptimizerConfig:
    def __init__(self, lr) -> None:
        if lr is None or not math.isfinite(lr):
            raise ValueError("learning rate must be a finite number")
        if lr < 0:
            raise ValueError("learning rate must greater or equal than zero.") 
        
        self.lr = lr