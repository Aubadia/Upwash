import time
import math
from contextlib import contextmanager, nullcontext

import torch

from upwash.core.state import TrainingState


class _LoggingInterval:
    """Accumulate training metrics and exclude non-training work from timing"""

    def __init__(self, *, timed: bool):
        self.timed = timed
        self.reset()

    def reset(self):
        self.loss = 0.0
        self.steps = 0
        self.tokens = 0
        self.step_metrics = {}
        self.duration = 0.0
        self.excluded_time = 0.0
        self.started_at = time.perf_counter() if self.timed else 0.0

    def record(self, loss, step_metrics, tokens):
        if self.timed:
            self.duration = time.perf_counter() - self.started_at - self.excluded_time
        self.loss += loss.detach()
        self.steps += 1
        self.tokens += tokens
        self.step_metrics = step_metrics

    def metrics(self, total_tokens):
        metrics = {"train/loss": self.loss.item() / self.steps, **self.step_metrics}
        metrics["train/grad_norm"] = float(metrics["train/grad_norm"])
        if self.timed:
            metrics["train/total_tokens_seen"] = int(total_tokens)
            metrics["train/tokens_per_second"] = (
                float(self.tokens) / self.duration if self.duration > 0 else 0.0
            )
        return metrics

    @contextmanager
    def exclude_time(self):
        """Exclude evaluation and logging time"""
        if not self.timed:
            yield
            return
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.excluded_time += time.perf_counter() - started_at


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer,
        lr_scheduler=None,
        logger=None,
        logging_steps=None,
        count_tokens=False,
        pad_token_id=None,
        eval_steps=None,
        precision="fp32",
        eval_perplexity=False
    ):
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
        self.state = TrainingState()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.logging_steps = logging_steps
        self.count_tokens = count_tokens
        self.pad_token_id = pad_token_id
        self.eval_steps = eval_steps
        self._total_tokens_seen = 0
        self.precision = precision
        self.eval_perplexity = eval_perplexity

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = self.model(inputs)
        return self.loss_fn(predictions, targets)

    def train(self, batches, *, eval_batches=None) -> TrainingState:
        self.model.train()
        self.state.set_state("global_step", 0)

        logging_enabled = self.logger is not None and self.logging_steps is not None
        eval_enabled = self.eval_steps is not None and eval_batches is not None

        interval = _LoggingInterval(timed=logging_enabled and self.count_tokens)
        steps = 0
        last_eval_step = 0
        pending_metrics = {}

        for inputs, targets in batches:
            # Log without excluded time
            self._log_metrics(pending_metrics, steps, interval)
            loss, step_metrics = self._training_step(inputs, targets, logging_enabled)
            steps += 1
            tokens = self._record_tokens(inputs)

            if logging_enabled:
                interval.record(loss, step_metrics, tokens)

            interval_complete = logging_enabled and interval.steps == self.logging_steps
            pending_metrics = (
                interval.metrics(self._total_tokens_seen) if interval_complete else {}
            )

            if eval_enabled and steps % self.eval_steps == 0:
                pending_metrics.update(self._evaluation_metrics(eval_batches, interval))
                last_eval_step = steps

            # Defer emission so final evaluation can join this step's event
            if interval_complete:
                interval.reset()

        if steps == 0:
            raise ValueError("batches cannot be empty")
        self.state.set_state("global_step", steps)

        if eval_enabled and last_eval_step != steps:
            pending_metrics.update(self._evaluation_metrics(eval_batches, interval))
        if logging_enabled and interval.steps:
            pending_metrics.update(interval.metrics(self._total_tokens_seen))
        self._log_metrics(pending_metrics, steps, interval)
        return self.state
    
    def _to_model_device(self, inputs, targets):
        device = next(self.model.parameters()).device
        return inputs.to(device), targets.to(device)
    
    def _autocast(self):
        if self.precision == "bf16":
            device = next(self.model.parameters()).device
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        else:
            return nullcontext()

    def _training_step(self, inputs, targets, logging_enabled):
        self.optimizer.zero_grad()
        inputs, targets = self._to_model_device(inputs, targets)
        with self._autocast():
            loss = self.compute_loss(inputs, targets)
        loss.backward()

        metrics = {}
        if logging_enabled:
            metrics["train/learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
            grad_norms = [
                parameter.grad.detach().norm(2)
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            metrics["train/grad_norm"] = (
                torch.stack(grad_norms).norm(2) if grad_norms else 0.0
            )

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step() 
        
        return loss, metrics

    def _record_tokens(self, inputs):
        if not self.count_tokens:
            return 0
        tokens = (
            inputs.numel()
            if self.pad_token_id is None
            else torch.count_nonzero(inputs != self.pad_token_id)
        )
        self._total_tokens_seen += tokens
        return tokens

    def _evaluation_metrics(self, batches, interval):
        with interval.exclude_time():
            loss = self.evaluate(batches)["loss"]
        metrics = {"eval/loss": loss}
        if self.eval_perplexity:
            try:
                metrics["eval/perplexity"] = math.exp(loss)
            except OverflowError:
                metrics["eval/perplexity"] = math.inf
        return metrics

    def _log_metrics(self, metrics, step, interval):
        """Wrapper function to log without excluded time"""
        if metrics and self.logger is not None:
            with interval.exclude_time():
                self.logger.log(metrics, step=step)

    def evaluate(self, batches) -> dict:
        previous_training = self.model.training
        total_loss = 0.0
        total_targets = 0

        try:
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in batches:
                    inputs, targets = self._to_model_device(inputs, targets)
                    with self._autocast():
                        loss = self.compute_loss(inputs, targets)
                    num_targets = targets.numel()
                    total_loss += loss.item() * num_targets
                    total_targets += num_targets

            if total_targets == 0:
                raise ValueError("num_targets is 0")
        finally:
            self.model.train(previous_training)

        return {"loss": total_loss / total_targets, "num_targets": total_targets}
