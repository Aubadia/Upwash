"""Run the raw DCLM v1 baseline on one eight-GPU Slurm node."""

import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from upwash.adaptor import build_llama_model, causal_lm_loss
from upwash.components.optimizers.combined import CombinedOptimizer
from upwash.config import ModelConfig, OptimizerConfig, RunConfig, ScheduleConfig, TrainingConfig
from upwash.core.trainer import TrainingProgress
from upwash.train import RunHooks, train
from upwash.prep.decontam import ContaminationIndex
from upwash.prep.pack import encode_document

SEED = 42
SEQ_LEN = 4096
GLOBAL_BATCH = 32
WORLD_SIZE = 8
TOKENS_PER_STEP = SEQ_LEN * GLOBAL_BATCH
TOTAL_STEPS = 30_000_000_000 // TOKENS_PER_STEP
CHECKPOINT_EVERY = 5000
PREFETCH_BATCHES = 16
DCLM_URI = "s3://flock-datasets/dclm"
PALOMA_URI = "s3://flock-datasets/paloma"
DATASET_VERSION = 1
TOKENIZER = "meta-llama/Llama-3.2-1B"
TOKENIZER_REV = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
BOS, EOS = 128000, 128001
MODEL = ModelConfig(128256, 6, 768, 3072, 12, 3, SEQ_LEN, 500_000.0, 1e-5, 0.1, 1.0)
SCHEDULE = ScheduleConfig(0.0005, 0.4, "power", 1.25)
OPTIMIZER = OptimizerConfig(0.0017, 0.9, 0.9, 0.99, 1.6, 0.0175, fused_adamw=True)
LEGACY_MUON_SCALES = {"muon_main": 0.6547, "muon_down": 0.7970}
ADAM_DECAY_CONSTANT = 2.63e-5
ROOT = Path("/work/asciibase64/upwash/dclm-150m")


def build_model(config=MODEL):
    model = build_llama_model(config)
    scale = 1 / config.n_layers

    def attention_scale(_module, _inputs, output):
        hidden, *other = output
        return (hidden * scale, *other)

    def mlp_scale(_module, _inputs, output):
        return output * scale

    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(attention_scale)
        layer.mlp.register_forward_hook(mlp_scale)
    return model


def split_parameters(model):
    matrices = []
    adam_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim != 2 or name.endswith(("embed_tokens.weight", "lm_head.weight")):
            adam_parameters.append(param)
        else:
            matrices.append((name, param))
    return matrices, adam_parameters


class IndependentDecayOptimizer(CombinedOptimizer):
    @torch.no_grad()
    def step(self):
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                if group["lr"] and group["independent_weight_decay"]:
                    # The common WSD factor scales both the update and c / width.
                    factor = group["lr"] / group["peak_lr"]
                    params = [param for param in group["params"] if param.grad is not None]
                    if params:
                        torch._foreach_mul_(params, 1 - factor * group["independent_weight_decay"])
            optimizer.step()


def build_optimizer(model, *, config, model_config=MODEL):
    matrices, adam_parameters = split_parameters(model)
    width = model_config.d_model
    adam_lr = config.lr * config.adam_lr_multiplier
    muon = torch.optim.Muon(
        [{
            "params": [param for _, param in matrices],
            "label": "muon",
            "peak_lr": config.lr,
            "independent_weight_decay": config.weight_decay / width,
        }],
        lr=config.lr, momentum=config.muon_beta, adjust_lr_fn="spectral_unclamped",
        weight_decay=0.0,
    )
    adam = torch.optim.AdamW(
        [{
            "params": adam_parameters,
            "label": "adamw",
            "peak_lr": adam_lr,
            "independent_weight_decay": ADAM_DECAY_CONSTANT / width,
        }],
        lr=adam_lr, betas=(config.adam_beta1, config.adam_beta2), weight_decay=0.0,
        fused=config.fused_adamw,
    )
    return IndependentDecayOptimizer([muon, adam])


def build_legacy_optimizer(model, *, config):
    """Reconstruct the optimizer groups for existing baseline checkpoints."""
    matrices, adam_parameters = split_parameters(model)
    groups = {}
    for name, param in matrices:
        label = "muon_down" if name.endswith("down_proj.weight") else "muon_main"
        # Torch's original Muon scales the gradient update by this factor and
        # uses the unadjusted LR for decoupled weight decay.
        adjustment = math.sqrt(max(1, param.shape[0] / param.shape[1]))
        group = groups.setdefault((label, adjustment), {
            "params": [],
            "label": label,
            "adjustment": adjustment,
            "lr": config.lr * LEGACY_MUON_SCALES[label] / adjustment,
            "weight_decay": config.weight_decay * adjustment,
        })
        group["params"].append(param)

    muon = torch.optim.Muon(
        list(groups.values()), momentum=config.muon_beta, adjust_lr_fn="original",
    )
    adam = torch.optim.AdamW(
        [{"params": adam_parameters, "label": "adamw", "adjustment": 1.0}],
        lr=config.lr * config.adam_lr_multiplier, betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=ADAM_DECAY_CONSTANT, fused=config.fused_adamw,
    )
    return CombinedOptimizer([muon, adam])


def packed_batches(documents, *, cursor, seq_len=SEQ_LEN, batch_size=GLOBAL_BATCH):
    """Yield complete global batches and a cursor at the next window's first token."""
    windows = []
    window = []
    _, resume_row, resume_offset = cursor
    for fragment, row, ids in documents:
        offset = resume_offset if (fragment, row) == (cursor[0], resume_row) else 0
        if offset >= len(ids):
            raise ValueError(f"Invalid token offset {offset} at fragment {fragment}, row {row}")
        while offset < len(ids):
            stop = min(len(ids), offset + seq_len + 1 - len(window))
            window.extend(ids[offset:stop])
            offset = stop
            if len(window) == seq_len + 1:
                windows.append(window)
                window = window[-1:]
                if len(windows) == batch_size:
                    yield np.asarray(windows, dtype=np.int32), (fragment, row, offset - 1)
                    windows.clear()


def open_sources():
    import lance
    from flock.hub import s3

    s3.get_process(read_only=True)
    return (
        lance.dataset(DCLM_URI, version=DATASET_VERSION),
        lance.dataset(PALOMA_URI, version=DATASET_VERSION),
    )


def load_tokenizer():
    from huggingface_hub import try_to_load_from_cache, hf_hub_download
    from tokenizers import Tokenizer

    path = try_to_load_from_cache(
        TOKENIZER, "tokenizer.json", revision=TOKENIZER_REV,
        cache_dir=os.environ["DCLM_EXISTING_HF_CACHE"],
    )
    if not isinstance(path, str):
        path = hf_hub_download(TOKENIZER, "tokenizer.json", revision=TOKENIZER_REV)
    tokenizer = Tokenizer.from_file(path)
    if (tokenizer.token_to_id("<|begin_of_text|>"), tokenizer.token_to_id("<|end_of_text|>")) != (BOS, EOS):
        raise ValueError("Unexpected Llama 3.2 special-token IDs")
    return tokenizer


def paloma_data(dataset):
    validation = []

    def texts():
        for batch in dataset.to_batches(columns=["_hf_split", "text"], batch_size=1024):
            for split, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
                if split in ("validation", "test") and text:
                    if split == "validation" and len(validation) < 128:
                        validation.append(text)
                    yield text

    index = ContaminationIndex.from_texts(texts())
    if not validation or not len(index):
        raise RuntimeError("Paloma v1 produced no validation text or decontamination hashes")
    return index, validation


def encoded_documents(dataset, tokenizer, index, cursor):
    fragments = sorted(dataset.get_fragments(), key=lambda item: item.fragment_id)
    random.Random(SEED).shuffle(fragments)
    resume_fragment, resume_row, _ = cursor
    for fragment_index, fragment in enumerate(fragments):
        if fragment_index < resume_fragment:
            continue
        start = resume_row if fragment_index == resume_fragment else 0
        raw_row = 0
        for batch in fragment.to_batches(columns=["text"], batch_size=1024):
            texts = batch.column(0).to_pylist()
            accepted = [
                (raw_row + offset, text)
                for offset, text in enumerate(texts)
                if raw_row + offset >= start and text and not index.is_contaminated(text)
            ]
            raw_row += len(texts)
            ids = tokenizer.encode_batch(
                [text for _, text in accepted], add_special_tokens=False,
            )
            for (row, _), encoded in zip(accepted, ids, strict=True):
                yield fragment_index, row, encode_document(encoded.ids, list, bos_id=BOS, eos_id=EOS)


class DistributedBatches:
    def __init__(self, runtime, source, *, trace=False):
        self.runtime = runtime
        self.cursor = None
        self.trace = trace
        self.first_hash = None
        self.last_hash = None
        self.output = torch.empty(
            (GLOBAL_BATCH // WORLD_SIZE, SEQ_LEN + 1), dtype=torch.int32, device=runtime.device,
        )
        self.available = torch.empty((), dtype=torch.int32, device=runtime.device)
        self.worker = ThreadPoolExecutor(max_workers=1) if runtime.is_main else None
        self.source = source
        self.prefetched = deque(
            self.worker.submit(next, source) for _ in range(PREFETCH_BATCHES)
        ) if self.worker else deque()
        self.read_wait_seconds = 0.0

    def peek(self):
        return self.prefetched[0].result()

    def take(self, steps):
        for _ in range(steps):
            chunks = None
            error = None
            if self.runtime.is_main:
                try:
                    started = time.perf_counter()
                    windows, self.cursor = self.prefetched.popleft().result()
                    self.read_wait_seconds += time.perf_counter() - started
                    self.prefetched.append(self.worker.submit(next, self.source))
                    if self.trace:
                        self.last_hash = hashlib.sha256(windows.tobytes()).hexdigest()
                        if self.first_hash is None:
                            self.first_hash = self.last_hash
                    chunks = list(torch.from_numpy(windows).to(self.runtime.device).chunk(WORLD_SIZE))
                except Exception as exc:
                    error = exc
                self.available.fill_(error is None)
            dist.broadcast(self.available, src=0)
            if not self.available.item():
                raise RuntimeError("DCLM stream could not produce the next complete batch") from error
            dist.scatter(self.output, scatter_list=chunks, src=0)
            tokens = self.output.long()
            yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    def close(self, *, wait=True):
        if self.worker is not None:
            self.worker.shutdown(wait=wait, cancel_futures=True)


def evaluate(model, tokenizer, validation, device):
    loss_sum = 0.0
    tokens = 0
    nbytes = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for text in validation:
                ids = [BOS] + tokenizer.encode(text, add_special_tokens=False).ids
                nbytes += len(text.encode("utf-8"))
                for start in range(0, len(ids) - 1, SEQ_LEN):
                    window = torch.tensor(ids[start:start + SEQ_LEN + 1], device=device).unsqueeze(0)
                    targets = window[:, 1:]
                    loss_sum += causal_lm_loss(model(window[:, :-1]), targets).item() * targets.numel()
                    tokens += targets.numel()
    finally:
        model.train(was_training)
    if not tokens or not nbytes or not math.isfinite(loss_sum):
        raise RuntimeError("Paloma validation loss is unavailable or non-finite")
    return {"eval/loss": loss_sum / tokens, "eval/bpb": loss_sum / (math.log(2) * nbytes),
            "eval/tokens": tokens, "eval/bytes": nbytes}


def metadata(attention, *, legacy_optimizer=False):
    if legacy_optimizer:
        recipe = {"muon_lr_scales": LEGACY_MUON_SCALES, "adam_weight_decay": ADAM_DECAY_CONSTANT}
    else:
        recipe = {
            "muon_adjust_lr_fn": "spectral_unclamped",
            "independent_weight_decay": {
                "width": MODEL.d_model, "schedule": "wsd",
                "muon_c": OPTIMIZER.weight_decay, "adamw_c": ADAM_DECAY_CONSTANT,
                "muon_peak": OPTIMIZER.weight_decay / MODEL.d_model,
                "adamw_peak": ADAM_DECAY_CONSTANT / MODEL.d_model,
            },
        }
    return {
        "model": MODEL.to_dict(), "residual_scale": 1 / MODEL.n_layers,
        "optimizer": OPTIMIZER.to_dict(), **recipe, "schedule": SCHEDULE.to_dict(),
        "precision": "bf16", "grad_clip": 1.0, "seed": SEED,
        "attention": attention, "compile_scope": "objective",
        "world_size": WORLD_SIZE, "global_batch": GLOBAL_BATCH,
        "tokens_per_step": TOKENS_PER_STEP, "planned_steps": TOTAL_STEPS,
        "planned_tokens": TOTAL_STEPS * TOKENS_PER_STEP,
        "dclm": [DCLM_URI, DATASET_VERSION], "paloma": [PALOMA_URI, DATASET_VERSION],
        "validation": "first 128 nonempty validation texts in Paloma v1 scan order",
        "tokenizer": [TOKENIZER, TOKENIZER_REV],
        "torch": str(torch.__version__),
    }


def save_checkpoint(path, *, model, trainer, runtime, config, progress, batches):
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state().cpu()}
    states = [None] * WORLD_SIZE if runtime.is_main else None
    dist.gather_object(rng, states, dst=0)
    if runtime.is_main:
        next_hash = None
        if batches.trace:
            next_windows, _ = batches.peek()
            next_hash = hashlib.sha256(next_windows.tobytes()).hexdigest()
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save({
                "model": model.state_dict(),
                "optimizers": [item.state_dict() for item in trainer.optimizer.optimizers],
                "scheduler": trainer.lr_scheduler.state_dict(),
                "step": progress.step, "tokens": progress.tokens,
                "cursor": batches.cursor, "rng": states, "config": config,
                "next_hash": next_hash,
            }, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    dist.barrier()


def learning_rates(optimizer):
    return {
        f"train/learning_rate/{group['label']}": group["lr"] * group.get("adjustment", 1.0)
        for group in optimizer.param_groups
    }


class RunLogger:
    def __init__(self, path, trainer):
        self.path = path
        self.scheduler = trainer.lr_scheduler
        self.peak_lrs = {
            f"train/learning_rate/{group['label']}": lr * group.get("adjustment", 1.0)
            for group, lr in zip(trainer.optimizer.param_groups, self.scheduler.base_lrs, strict=True)
        }

    def discard_after(self, step):
        if not self.path.exists():
            return
        lines = self.path.read_text().splitlines(keepends=True)
        retained = [line for line in lines if json.loads(line)["step"] <= step]
        if len(lines) != len(retained):
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text("".join(retained))
            os.replace(temporary, self.path)

    def log(self, metrics, *, step):
        if "train/loss" in metrics:
            factor = self.scheduler.factor_at(step)
            metrics = {**metrics, **{name: lr * factor for name, lr in self.peak_lrs.items()}}
            if not math.isfinite(metrics["train/loss"]):
                raise RuntimeError(f"Non-finite training loss at step {step}")
        line = json.dumps({"step": step, **metrics}, allow_nan=False)
        with self.path.open("a") as output:
            output.write(line + "\n")
        print(line, flush=True)


class DCLMRun:
    """Provide experiment inputs and checkpoint handling to upwash.train.train."""

    def __init__(self, args):
        self.args = args
        self.limit = args.stop_after
        recorded = json.loads((args.run_dir / "config.json").read_text()) if args.resume else {}
        self.legacy_optimizer = "muon_lr_scales" in recorded
        self.config = metadata(args.attention, legacy_optimizer=self.legacy_optimizer)
        self.checkpoint = args.run_dir / "latest.pt"
        self.progress = TrainingProgress()
        self.started = time.monotonic()

    @contextmanager
    def batches(self, runtime):
        self.runtime = runtime
        if runtime.world_size != WORLD_SIZE:
            raise RuntimeError("Run with torchrun and eight GPUs")
        self.saved = None
        cursor = (0, 0, 0)
        if self.args.resume:
            self.saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            if self.saved["config"] != self.config:
                raise ValueError("Checkpoint configuration differs from the requested run")
            step = self.saved["step"]
            self.progress = TrainingProgress(step, self.saved.get("tokens", step * TOKENS_PER_STEP))
            cursor = self.saved["cursor"]
        if self.progress.step >= self.limit:
            raise ValueError(f"Checkpoint is already at step {self.progress.step}, requested stop {self.limit}")

        source = None
        if runtime.is_main:
            if not self.args.resume:
                self.args.run_dir.mkdir(parents=False, exist_ok=False)
                (self.args.run_dir / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
            dclm, paloma = open_sources()
            self.tokenizer = load_tokenizer()
            index, self.validation = paloma_data(paloma)
            print(f"READY dclm_version={dclm.version} paloma_version={paloma.version} "
                  f"decontamination_hashes={len(index)} startup_seconds={time.monotonic() - self.started:.2f}",
                  flush=True)
            source = packed_batches(encoded_documents(dclm, self.tokenizer, index, cursor), cursor=cursor)

        self.stream = DistributedBatches(runtime, source, trace=self.limit < TOTAL_STEPS)
        try:
            yield self.stream.take(self.limit - self.progress.step)
        except BaseException:
            traceback.print_exc()
            self.stream.close(wait=False)
            raise
        else:
            self.stream.close()

    def prepare(self, runtime, model, trainer):
        self.model = model
        self.trainer = trainer
        if self.saved is not None:
            model.load_state_dict(self.saved["model"])
            for optimizer, state in zip(trainer.optimizer.optimizers, self.saved["optimizers"], strict=True):
                optimizer.load_state_dict(state)
            trainer.lr_scheduler.load_state_dict(
                self.saved.get("scheduler", {"update": self.progress.step + 1}),
            )
            rng = self.saved["rng"][runtime.rank]
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"])
            if runtime.is_main and (expected := self.saved.get("next_hash")):
                next_windows, _ = self.stream.peek()
                if hashlib.sha256(next_windows.tobytes()).hexdigest() != expected:
                    raise RuntimeError("Restored stream disagrees with the first batch after the checkpoint")
                print(f"RESUME batch_hash={expected} lr={learning_rates(trainer.optimizer)}", flush=True)
            self.saved = None

        if runtime.is_main:
            trainer.logger = RunLogger(self.args.run_dir / "metrics.jsonl", trainer)
            if self.args.resume:
                trainer.logger.discard_after(self.progress.step)
        self.initial_weight = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        self.training_started = time.monotonic()
        return self.progress

    def after_step(self, progress):
        step = progress.step
        final_step = step == self.limit
        if final_step:
            if torch.equal(self.initial_weight, self.model.model.layers[0].self_attn.q_proj.weight):
                raise RuntimeError("Training did not complete real parameter updates")
            torch.cuda.synchronize()
            if self.runtime.is_main:
                duration = time.monotonic() - self.training_started
                trained_tokens = progress.tokens - self.progress.tokens
                print(f"SEGMENT start={self.progress.step} end={step} seconds={duration:.3f} "
                      f"tokens_per_second={trained_tokens / duration:.0f} "
                      f"stream_wait_seconds={self.stream.read_wait_seconds:.3f} "
                      f"first_batch={self.stream.first_hash} last_batch={self.stream.last_hash}", flush=True)

        if not final_step and step % CHECKPOINT_EVERY:
            return None

        save_checkpoint(
            self.checkpoint, model=self.model, trainer=self.trainer, runtime=self.runtime,
            config=self.config, progress=progress, batches=self.stream,
        )
        metrics = (
            evaluate(self.model, self.tokenizer, self.validation, self.runtime.device)
            if self.runtime.is_main else None
        )
        dist.barrier()
        return metrics

    def run(self):
        config = RunConfig(
            model=MODEL, train_data=None,
            training=TrainingConfig(OPTIMIZER, logging_steps=100, precision="bf16", max_grad_norm=1.0),
            schedule=SCHEDULE, batch_size=GLOBAL_BATCH // WORLD_SIZE, device="cuda",
            seed=SEED, compile_scope="objective", attention_implementation=self.args.attention,
        )
        optimizer_factory = (
            build_legacy_optimizer if self.legacy_optimizer else partial(build_optimizer, model_config=config.model)
        )
        result = train(config, hooks=RunHooks(
            batches=self.batches, build_model=build_model, build_optimizer=optimizer_factory,
            prepare=self.prepare, after_step=self.after_step, schedule_steps=TOTAL_STEPS,
        ))
        if result.state.global_step != self.limit:
            raise RuntimeError("Training did not reach the requested endpoint")
        if self.runtime.is_main and self.limit == TOTAL_STEPS:
            os.replace(self.checkpoint, self.args.run_dir / "final.pt")
            print(f"COMPLETE steps={self.limit} tokens={self.limit * TOKENS_PER_STEP}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after", type=int, default=TOTAL_STEPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--attention", choices=["upwash_fa3", "sdpa"], default="upwash_fa3")
    args = parser.parse_args()
    if args.preflight:
        dclm, paloma = open_sources()
        tokenizer = load_tokenizer()
        for dataset in (dclm, paloma):
            preview = next(dataset.to_batches(columns=["text"], batch_size=1))
            print(f"OPEN {dataset.uri} version={dataset.version} schema={dataset.schema.names} "
                  f"preview_tokens={len(tokenizer.encode(preview.column(0)[0].as_py()).ids)}")
        return
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("--run-id must be a single directory name")
    if not 1 <= args.stop_after <= TOTAL_STEPS:
        parser.error(f"--stop-after must be between 1 and {TOTAL_STEPS}")
    args.run_dir = ROOT / "runs" / args.run_id
    DCLMRun(args).run()


if __name__ == "__main__":
    main()
