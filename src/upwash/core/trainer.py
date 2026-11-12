import torch

from upwash.core.state import TrainingState

class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn, optimizer, logger=None):
        self.state = TrainingState()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.logger = logger

    def compute_loss(
            self, 
            inputs: torch.Tensor, 
            targets: torch.Tensor
    ) -> torch.Tensor:
        out = self.model(inputs)
        return self.loss_fn(out, targets)

    def train(self, batches) -> TrainingState:
        self.model.train() 
       
        self.state.set_state("global_step", 0)
        steps = 0
        
        for input, target in batches:
            self.optimizer.zero_grad()
            loss = self.compute_loss(input, target)
            loss.backward()
            self.optimizer.step()
            steps = steps + 1

        if steps == 0:
            raise ValueError("batches cannot be empty")
        
        self.state.set_state("global_step", steps) 
        
        # logger.log() when needed

        return self.state

    def evaluate(self, batches) -> dict:
        prev_training = self.model.training
        total_loss = 0.0
        total_num = 0

        try:
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in batches:
                    loss = self.compute_loss(inputs, targets)
                    # Weighted loss, since the last shard may be truncated
                    total_loss += loss.item() * targets.numel()
                    total_num += targets.numel()
        
            if total_num == 0:
                raise ValueError("num_targets is 0")
            eval_loss = total_loss / total_num
        finally:
            self.model.train(prev_training)

        return {"loss": eval_loss, "num_targets": total_num}