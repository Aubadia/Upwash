import torch

from upwash.core.state import TrainingState

class Trainer:
    def __init__(
            self, 
            model: torch.nn.Module, 
            loss_fn, 
            optimizer, 
            logger=None,
            logging_steps=None,
        ):
        self.state = TrainingState()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.logger = logger
        self.logging_steps=logging_steps

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
        logging_enabled = self.logger is not None and self.logging_steps is not None
        interval_loss = 0.0
        interval_steps = 0
        
        for input, target in batches:
            self.optimizer.zero_grad()
            loss = self.compute_loss(input, target)
            loss.backward()

            if logging_enabled:
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                grad_norms = [
                    parameter.grad.detach().norm(2)
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                grad_norm = torch.stack(grad_norms).norm(2) if grad_norms else 0.0
                
            self.optimizer.step()
            steps = steps + 1

            if logging_enabled:
                interval_loss += loss.detach()
                interval_steps += 1
                if interval_steps == self.logging_steps:
                    self.logger.log(
                        {
                            "train/loss": interval_loss.item() / interval_steps,
                            "train/learning_rate": learning_rate,
                            "train/grad_norm": float(grad_norm),
                        },
                        step=steps,
                    )
                    interval_loss = 0.0
                    interval_steps = 0

        if steps == 0:
            raise ValueError("batches cannot be empty")
        
        self.state.set_state("global_step", steps) 

        # Log additional information for the last step
        if logging_enabled and interval_steps:
            self.logger.log(
                {
                    "train/loss": interval_loss.item() / interval_steps,
                    "train/learning_rate": learning_rate,
                    "train/grad_norm": float(grad_norm),
                },
                step=steps,
            )

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
