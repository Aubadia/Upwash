from torch.optim import Optimizer

class CombinedOptimizer:
    def __init__(self, optimizers:list[Optimizer]) -> None:
        self.optimizers = optimizers
    
    def step(self):
        for optim in self.optimizers:
            optim.step()
    
    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()
    
    @property
    def param_groups(self):
        return [
            group
            for optim in self.optimizers
            for group in optim.param_groups
        ]