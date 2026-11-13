"""Slurm-only real Llama smoke test with synthetic tokens and real Flock logging.

Run with scripts/test_train.sbatch. Synthetic train/dev shards are created in
pytest's temporary directory under /work/$USER. No dataset downloads required.
"""

import importlib
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch



def test_train_real_llama_on_cuda_with_scheduled_and_final_evaluation(
    tmp_path, make_optimizer_config, monkeypatch,
):
    assert os.environ.get("WANDB_MODE") == "online", "Flock test requires WANDB_MODE=online"
    assert os.environ.get("WANDB_BASE_URL"), "Set WANDB_BASE_URL to the Flock API host"
    assert os.environ.get("WANDB_API_KEY"), "Set WANDB_API_KEY to your Flock credential"

    from transformers import LlamaConfig, LlamaForCausalLM

    from upwash.builder import build_dataset
    from upwash.config import DataConfig, LoggingConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig
    from upwash.core.logger import FlockLogger
    from upwash.core.trainer import Trainer

    assert torch.cuda.is_available(), "This test requires a CUDA allocation via sbatch"
    work_root = (Path("/work") / os.environ["USER"]).resolve()
    assert tmp_path.resolve().is_relative_to(work_root), (
        "Synthetic data must be stored under /work/$USER; use scripts/test_train.sbatch"
    )
    vocab_size = 32
    train_dir, dev_dir = tmp_path / "train", tmp_path / "dev"
    train_dir.mkdir()
    dev_dir.mkdir()
    np.save(train_dir / "tokens.npy", np.arange(4097, dtype=np.int32) % vocab_size)
    np.save(dev_dir / "tokens.npy", (np.arange(1025, dtype=np.int32) + 1) % vocab_size)
    train_data = DataConfig(str(train_dir), "tokens.npy", "npy", 256, 4096)
    dev_data = DataConfig(str(dev_dir), "tokens.npy", "npy", 256, 1024)
    train_dataset, dev_dataset = build_dataset(train_data), build_dataset(dev_data)
    assert len(train_dataset) == 16
    assert len(dev_dataset) == 4

    # First establish real HF forward/backward on the GPU with synthetic tokens.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        reference = LlamaForCausalLM(LlamaConfig(
            vocab_size=vocab_size, hidden_size=128, intermediate_size=384,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, tie_word_embeddings=True,
            attention_dropout=0.0,
        )).to("cuda")
        inputs, targets = train_dataset[0]
        inputs, targets = inputs[None].to("cuda"), targets[None].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = reference(inputs, use_cache=False).logits
            reference_loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets)
        reference_loss.backward()
        assert torch.isfinite(reference_loss)
        gradient = reference.model.embed_tokens.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
        print(f"Synthetic HF reference loss: {reference_loss.item():.6f}")
    del reference, logits, reference_loss, gradient

    from upwash.train import TrainResult, train

    entry = importlib.import_module("upwash.train")
    original_build_model = entry.build_llama_model
    observed_names = ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight")
    initial_parameters = {}

    def record_initial_model(model_config):
        model = original_build_model(model_config)
        model.config.use_cache = False
        initial_parameters.update({
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if name in observed_names
        })
        return model

    monkeypatch.setattr(entry, "build_llama_model", record_initial_model)
    config = RunConfig(
        model=ModelConfig(
            vocab_size=vocab_size, n_layers=2, d_model=128, d_ff=384,
            n_heads=4, n_kv_heads=2, max_seq_len=256,
            rope_theta=500_000.0, norm_eps=1e-5,
            embedding_std=0.1, output_std_coeff=1.0,
        ),
        train_data=train_data,
        eval_data=dev_data,
        training=TrainingConfig(
            optimizer=make_optimizer_config(), logging_steps=1, eval_steps=3, precision="bf16",
        ),
        schedule=ScheduleConfig(
            warmup_ratio=0.0, decay_ratio=0.0, decay_shape="power", decay_power=1.0,
        ),
        batch_size=2, eval_batch_size=2, device="cuda", seed=17,
        logging=LoggingConfig(
            project=os.environ.get("FLOCK_PROJECT", "lab6-predictable-scaling"),
            name=os.environ["FLOCK_RUN_NAME"],
        ),
    )
    logged_events = []
    original_log = FlockLogger.log

    def record_log(self, metrics, step, commit=True):
        original_log(self, metrics, step=step, commit=commit)
        logged_events.append((dict(metrics), step))

    # Observe real forwarding; initialization, delivery and finish are not mocked.
    monkeypatch.setattr(FlockLogger, "log", record_log)
    evaluations = []
    original_evaluate = Trainer.evaluate

    def record_evaluation(self, batches):
        result = original_evaluate(self, batches)
        evaluations.append(result)
        print(f"Synthetic dev evaluation {len(evaluations)}: {result}")
        return result

    monkeypatch.setattr(Trainer, "evaluate", record_evaluation)
    expected_steps = math.ceil(len(train_dataset) / config.batch_size)
    print(
        f"Synthetic train={train_data.num_tokens:,} tokens, "
        f"dev={dev_data.num_tokens:,} tokens, steps={expected_steps}"
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        result = train(config)

    assert isinstance(result, TrainResult)
    assert isinstance(result.model, LlamaForCausalLM)
    assert result.state.global_step == expected_steps
    assert all(p.device.type == "cuda" for p in result.model.parameters())
    assert all(torch.isfinite(p).all() for p in result.model.parameters())
    for name in observed_names:
        trained = dict(result.model.named_parameters())[name].detach().cpu()
        assert not torch.equal(trained, initial_parameters[name]), name
    expected_evals = math.ceil(expected_steps / config.training.eval_steps)
    assert len(evaluations) == expected_evals
    assert all(item["num_targets"] == dev_data.num_tokens for item in evaluations)
    assert all(math.isfinite(item["loss"]) for item in evaluations)
    assert [step for _, step in logged_events] == list(range(1, expected_steps + 1))
    assert logged_events[-1][0]["train/total_tokens_seen"] == train_data.num_tokens
    eval_events = [metrics for metrics, _ in logged_events if "eval/loss" in metrics]
    assert len(eval_events) == expected_evals
    for metrics, result in zip(eval_events, evaluations, strict=True):
        assert metrics["eval/loss"] == pytest.approx(result["loss"])
        assert metrics["eval/perplexity"] == pytest.approx(math.exp(result["loss"]))
