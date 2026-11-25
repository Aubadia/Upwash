"""Train an Upwash recipe with full Paloma evaluation at the endpoint."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
from tempfile import mkdtemp
import time

import numpy as np
import torch
import torch.distributed as dist

from scripts.datadecide_paloma import atomic_json, read_json
from scripts.train_dclm_150m import (
    BOS, CHECKPOINT_EVERY, EOS, ROOT, SEED, TOKENS_PER_STEP, TOTAL_STEPS,
    DCLMRun, evaluate_checkpoint,
    load_tokenizer, open_datasets, packed_batches, paloma_data,
)

THRESHOLD = 1.25
SOURCE_ROW_TOKENS = 8192
TOKENIZER_DIR = Path("/work/asciibase64/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/"
                     "snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08")


def native_documents(rows, *, eos, resume_offset=None):
    """Yield complete documents in one .npy file and their native offsets."""
    next_offset = None
    start = resume_offset
    pieces = []
    for offset, values in rows:
        values = np.asarray(values)
        if next_offset is not None and offset != next_offset:
            raise ValueError(f"Non-contiguous native rows: expected {next_offset}, got {offset}")
        next_offset = offset + len(values)
        begin = 0
        if resume_offset is not None:
            if not offset <= resume_offset < next_offset:
                raise ValueError(f"Native cursor {resume_offset} is outside its first row")
            begin = resume_offset - offset
            resume_offset = None
        for end in np.flatnonzero(values[begin:] == eos) + begin:
            if start is not None:
                if end > begin:
                    pieces.append(values[begin:end])
                if pieces:
                    yield start, np.concatenate(pieces).tolist()
            pieces.clear()
            begin = int(end) + 1
            start = offset + begin
        if start is not None and begin < len(values):
            pieces.append(values[begin:])


def file_rows(dataset, source, resume_offset):
    target = resume_offset if resume_offset is not None else 0
    columns = ["token_ids", "_hf_source_file", "source_token_offset", "source_token_count"]
    for fragment_id, start, end in source["fragments"]:
        if end <= target:
            continue
        fragment = dataset.get_fragment(fragment_id)
        skip = (max(start, target) - start) // SOURCE_ROW_TOKENS
        position = start + skip * SOURCE_ROW_TOKENS
        for batch in fragment.to_batches(columns=columns, offset=skip, batch_size=64):
            paths, offsets, sizes = (batch.column(i).to_pylist() for i in (1, 2, 3))
            for index, (path, offset, size) in enumerate(zip(paths, offsets, sizes, strict=True)):
                values = batch.column(0)[index].values.to_numpy(zero_copy_only=False)
                if (path != source["path"] or offset != position or len(values) != size
                        or size != min(SOURCE_ROW_TOKENS, end - position)):
                    raise ValueError(f"Lance manifest mismatch in fragment {fragment_id}, row offset {offset}")
                yield offset, values
                position += size
        if position != end:
            raise ValueError(f"Lance fragment {fragment_id} ended at {position}, expected {end}")


def component_documents(dataset, files, pointer, eos):
    if pointer is None:
        return
    start_file, offset = pointer
    for file_index in range(start_file, len(files)):
        resume_offset = offset if file_index == start_file and offset else None
        rows = file_rows(dataset, files[file_index], resume_offset)
        for native_offset, ids in native_documents(rows, eos=eos, resume_offset=resume_offset):
            yield file_index, native_offset, ids


def mix_documents(open_at, totals, state):
    """Merge components by the fraction of native tokens consumed."""
    pointers, consumed = state
    consumed = list(consumed)
    streams = [iter(open_at(index, point)) for index, point in enumerate(pointers)]
    upcoming = [next(stream, None) for stream in streams]
    while any(item is not None for item in upcoming):
        # The input order breaks ties deterministically.
        group = min((i for i, item in enumerate(upcoming) if item is not None),
                    key=lambda i: consumed[i] / totals[i])
        ids = upcoming[group][2]
        cursor = (tuple(item[:2] if item is not None else None for item in upcoming),
                  tuple(consumed))
        yield cursor, ids
        consumed[group] += len(ids) + 1
        upcoming[group] = next(streams[group], None)


def encoded_recipe(dataset, groups, native, tokenizer, index, cursor, eos):
    files = []
    for group_index, group in enumerate(groups):
        order = sorted(group["files"], key=lambda file: file["path"])
        random.Random(SEED + group_index).shuffle(order)
        files.append(order)

    def open_at(group, pointer):
        return component_documents(dataset, files[group], pointer, eos)

    start_state = (tuple((0, 0) for _ in groups), tuple(0 for _ in groups))
    documents = mix_documents(open_at, [group["tokens"] for group in groups], cursor[0] or start_state)
    while chunk := list(islice(documents, 512)):
        texts = native.decode_batch([ids for _, ids in chunk], skip_special_tokens=False)
        accepted = [(state, text) for (state, _), text in zip(chunk, texts, strict=True)
                    if text and not index.is_contaminated(text)]
        ids = tokenizer.encode_batch([text for _, text in accepted], add_special_tokens=False)
        for (state, _), tokens in zip(accepted, ids, strict=True):
            yield state, 0, [BOS, *tokens.ids, EOS]


def official_decision(local_bpb, at_horizon, official_bpb):
    local_pass = local_bpb <= THRESHOLD
    return local_pass or at_horizon, local_pass and official_bpb is not None and official_bpb <= THRESHOLD


def confirmed_metrics(run_dir, checkpoint, step, local, at_horizon):
    should_evaluate, _ = official_decision(local["eval/bpb"], at_horizon, None)
    if not should_evaluate:
        return local, None, False

    from scripts.eval_dclm_paloma import score

    directory = run_dir / "official" / f"step-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    pending = directory / "local.json"
    atomic_json(pending, {"sha256": digest, "metrics": local})
    summary_path = directory / "confirmed.json"
    if summary_path.exists():
        result = read_json(summary_path)
        sources = directory / result["attempt"] / "paloma-summary.json"
        official = read_json(sources)
    else:
        attempt = Path(mkdtemp(prefix="attempt-", dir=directory))
        try:
            official = score(
                checkpoint, TOKENIZER_DIR, attempt, limit=None, batch_tokens=8192,
                allow_training_checkpoint=True,
            )
        except Exception as error:
            atomic_json(attempt / "error.json", {"checkpoint_sha256": digest, "error": str(error)})
            raise
        sources = attempt / "paloma-summary.json"
    if (official["checkpoint_sha256"] != digest or official["step"] != step
            or official["total_tokens"] != step * TOKENS_PER_STEP
            or official["limit_per_task"] is not None or len(official["paloma_source_bpb"]) != 9
            or not math.isfinite(official["paloma_macro_bpb"])
            or not all(math.isfinite(value) for value in official["paloma_source_bpb"].values())):
        raise ValueError("Official evaluation does not match the saved checkpoint and full suite")
    if not summary_path.exists():
        atomic_json(summary_path, {"attempt": sources.parent.name, "sha256": digest})
    official_bpb = official["paloma_macro_bpb"]
    _, qualified = official_decision(local["eval/bpb"], at_horizon, official_bpb)
    metrics = {
        **local, "paloma/macro_bpb": official_bpb, "paloma/qualified": qualified,
        **{f"paloma/source/{key.removeprefix('paloma:')}": value
           for key, value in official["paloma_source_bpb"].items()},
    }
    return metrics, official, qualified


def finish(run_dir, checkpoint, step, tokens, metrics, official, qualified):
    final = run_dir / "final.pt"
    if final.is_file():
        if not final.samefile(checkpoint):
            raise RuntimeError("Final checkpoint differs from the checkpoint being confirmed")
    else:
        # Keep the recoverable latest alias without duplicating the checkpoint's bytes.
        os.link(checkpoint, final)
    result = {
        "step": step, "tokens": tokens, "qualified": qualified,
        "stop_reason": "paloma_threshold" if qualified and step < TOTAL_STEPS else "token_limit",
        "checkpoint": str(final), "checkpoint_sha256": official["checkpoint_sha256"],
        "metrics": metrics,
    }
    atomic_json(run_dir / "completion.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def merge_metrics(path, step, tokens, metrics):
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    events = [event for event in events if event["step"] <= step]
    event = {"step": step, "train/total_tokens_seen": tokens, **metrics}
    if events and events[-1]["step"] == step:
        events[-1].update(event)
    else:
        events.append(event)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events))
    os.replace(temporary, path)


def recover(args):
    run_dir = args.run_dir
    completion_path = run_dir / "completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        if args.early_stop or completion["step"] == TOTAL_STEPS:
            return 75
        if completion["stop_reason"] != "paloma_threshold":
            raise RuntimeError("Unrecognized recipe completion")
        final = run_dir / "final.pt"
        if final.exists() and not final.samefile(run_dir / "latest.pt"):
            raise RuntimeError("Early-stop checkpoint differs from the resume checkpoint")
        final.unlink(missing_ok=True)
        os.replace(completion_path, run_dir / f"early-completion-step-{completion['step']}.json")
    checkpoint = run_dir / "latest.pt"
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    step, tokens = saved["step"], saved["tokens"]
    if saved["config"] != RecipeRun(args).config:
        raise ValueError("Checkpoint configuration differs from the requested run")
    del saved
    path = run_dir / "metrics.jsonl"
    event = next((json.loads(line) for line in reversed(path.read_text().splitlines())
                  if json.loads(line)["step"] == step), {}) if path.exists() else {}
    if "eval/bpb" in event:
        local = {name: value for name, value in event.items() if name.startswith("eval/")}
    else:
        local = evaluate_checkpoint(checkpoint)
        local.pop("step")
    at_horizon = step == TOTAL_STEPS
    if not (at_horizon or args.early_stop):
        merge_metrics(path, step, tokens, local)
        print(f"RESUME full-horizon step={step} tokens={tokens}", flush=True)
        return 0
    metrics, official, qualified = confirmed_metrics(run_dir, checkpoint, step, local, at_horizon)
    merge_metrics(path, step, tokens, metrics)
    if qualified or step == TOTAL_STEPS:
        finish(run_dir, checkpoint, step, tokens, metrics, official, qualified)
        return 75
    return 0


class RecipeRun(DCLMRun):
    def __init__(self, args):
        super().__init__(args)
        manifest = args.manifest.read_bytes()
        self.manifest = json.loads(manifest)
        members = self.manifest["recipes"][args.recipe]
        self.groups = [self.manifest["groups"][member] for member in members]
        self.config.pop("dclm")
        self.config.update(
            recipe=args.recipe, dataset=self.manifest["dataset"],
            recipe_manifest_sha256=hashlib.sha256(manifest).hexdigest(), paloma_threshold=THRESHOLD,
        )
        self.stop = False
        self.final_metrics = None
        self.final_official = None

    def make_source(self, cursor):
        from tokenizers import Tokenizer

        dataset, paloma = open_datasets(self.manifest["dataset"], self.config["paloma"])
        native_config = self.manifest["native_tokenizer"]
        path = Path(native_config["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != native_config["sha256"]:
            raise ValueError("Native tokenizer differs from the recipe manifest")
        native = Tokenizer.from_file(str(path))
        if native.token_to_id("<|endoftext|>") != native_config["eos"]:
            raise ValueError("Unexpected native EOS ID")
        self.tokenizer = load_tokenizer()
        index, self.validation = paloma_data(paloma)
        print(f"READY recipe={self.args.recipe} decontamination_hashes={len(index)} "
              f"startup_seconds={time.monotonic() - self.started:.2f}", flush=True)
        documents = encoded_recipe(dataset, self.groups, native, self.tokenizer, index, cursor, native_config["eos"])
        return packed_batches(documents, cursor=cursor)

    @contextmanager
    def batches(self, runtime):
        with super().batches(runtime) as stream:
            self.confirm_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=45))

            def until_stop():
                while not self.stop:
                    batch = next(stream, None)
                    if batch is None:
                        return
                    yield batch

            yield until_stop()

    def after_step(self, progress):
        local = super().after_step(progress)
        at_horizon = progress.step == TOTAL_STEPS
        if not at_horizon and not self.args.early_stop:
            return local
        if progress.step != self.limit and progress.step % CHECKPOINT_EVERY:
            return None
        message = [None]
        metrics = local
        if self.runtime.is_main:
            try:
                metrics, official, qualified = confirmed_metrics(
                    self.args.run_dir, self.checkpoint, progress.step, local, at_horizon,
                )
                self.final_metrics, self.final_official = metrics, official
                message[0] = {"qualified": qualified}
            except Exception as error:
                message[0] = {"error": str(error)}
        dist.broadcast_object_list(message, src=0, group=self.confirm_group)
        if "error" in message[0]:
            raise RuntimeError(f"Paloma confirmation failed at step {progress.step}: {message[0]['error']}")
        self.stop = self.args.early_stop and message[0]["qualified"]
        return metrics

    def finalize(self, result):
        step = result.state.global_step
        if step != self.limit and not self.stop:
            raise RuntimeError("Unexpected training endpoint")
        if self.runtime.is_main and (self.stop or step == TOTAL_STEPS):
            finish(
                self.args.run_dir, self.checkpoint, step, step * self.config["tokens_per_step"],
                self.final_metrics, self.final_official, self.final_metrics["paloma/qualified"],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=TOTAL_STEPS)
    parser.add_argument("--early-stop", action="store_true",
                        help="Stop after both local and official Paloma BPB meet the threshold.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("Invalid run ID")
    if not 1 <= args.stop_after <= TOTAL_STEPS:
        parser.error("Invalid stop step")
    args.run_dir = ROOT / "runs" / args.run_id
    args.attention = "upwash_fa3"
    if args.recover:
        status = recover(args)
        if job_id := os.environ.get("SLURM_JOB_ID"):
            (args.run_dir / f".resume-{job_id}.ready").touch()
        raise SystemExit(status)
    RecipeRun(args).run()


if __name__ == "__main__":
    main()
