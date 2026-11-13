"""Explicit sbatch-only smoke test using local FineWeb-Edu token shards.

Datasets default to separate directories under /work/$USER.
Optional: FINEWEB_{TRAIN,DEV}_FORMAT (npy/bin), _PATTERN, and _TOKENS.
Defaults: npy training, bin dev, 4096 training tokens, 1024 dev tokens.
Logs to Flock using WANDB_BASE_URL/WANDB_API_KEY; run with scripts/test_train.sbatch.
"""

import importlib
import math
import os
from pathlib import Path

import pytest
import torch


def fineweb_data_config(split, default_format, default_tokens):
    from upwash.config import DataConfig

    prefix = f"FINEWEB_{split}"
    work_root = Path("/work") / os.environ["USER"]
    default_dir = "fineweb-edu-llama3" if split == "TRAIN" else "fineweb-edu-dev-tokens"
    root = Path(os.environ.get(f"{prefix}_ROOT", str(work_root / default_dir))).resolve()
    if not root.is_relative_to(work_root.resolve()):
        pytest.fail(f"{prefix}_ROOT must be stored under {work_root}")
    data_format = os.environ.get(f"{prefix}_FORMAT", default_format)
    pattern = os.environ.get(
        f"{prefix}_PATTERN", "**/tokens.npy" if data_format == "npy" else "**/*.bin",
    )
    return DataConfig(
        root=str(root), pattern=pattern, format=data_format, seq_len=256,
        num_tokens=int(os.environ.get(f"{prefix}_TOKENS", default_tokens)),
    )


def test_train_real_llama_on_cuda_with_scheduled_and_final_evaluation(
    make_optimizer_config, monkeypatch,
):
    assert os.environ.get("WANDB_MODE") == "online", "Flock test requires WANDB_MODE=online"
    assert os.environ.get("WANDB_BASE_URL"), "Set WANDB_BASE_URL to the Flock API host"
    assert os.environ.get("WANDB_API_KEY"), "Set WANDB_API_KEY to your Flock credential"

    from transformers import LlamaConfig, LlamaForCausalLM

    from upwash.builder import build_dataset
    from upwash.config import LoggingConfig, ModelConfig, RunConfig, ScheduleConfig, TrainingConfig
    from upwash.core.logger import FlockLogger
    from upwash.core.trainer import Trainer

    train_data = fineweb_data_config("TRAIN", "npy", 4096)
    dev_data = fineweb_data_config("DEV", "bin", 1024)
    for data in (train_data, dev_data):
        assert data.num_tokens % data.seq_len == 0, "Token budgets must contain whole sequences"
    training_paths = {p.resolve() for p in Path(train_data.root).glob(train_data.pattern)}
    dev_paths = {p.resolve() for p in Path(dev_data.root).glob(dev_data.pattern)}
    work_root = (Path("/work") / os.environ["USER"]).resolve()
    assert all(path.is_relative_to(work_root) for path in training_paths | dev_paths), (
        "Shard files, including symlink targets, must be stored under /work/$USER"
    )
    assert training_paths.isdisjoint(dev_paths), "Training and dev must use separate shard files"
    train_dataset, dev_dataset = build_dataset(train_data), build_dataset(dev_data)
    vocab_size = 128256  # Llama-3.2 tokenizer vocabulary, including special tokens
    for dataset in (train_dataset, dev_dataset):
        for inputs, targets in dataset:
            assert inputs.dtype == targets.dtype == torch.int64
            assert 0 <= inputs.min() and inputs.max() < vocab_size
            assert 0 <= targets.min() and targets.max() < vocab_size

    assert torch.cuda.is_available(), "This test requires a CUDA allocation via sbatch"
    # First establish real HF forward/backward on the GPU with actual FineWeb tokens.
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
        print(f"FineWeb HF reference loss: {reference_loss.item():.6f}")
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
        print(f"FineWeb dev evaluation {len(evaluations)}: {result}")
        return result

    monkeypatch.setattr(Trainer, "evaluate", record_evaluation)
    expected_steps = math.ceil(len(train_dataset) / config.batch_size)
    print(
        f"FineWeb train={train_data.num_tokens:,} tokens, "
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
