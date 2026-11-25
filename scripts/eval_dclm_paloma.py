"""Score a saved Upwash DCLM checkpoint with Flock's pinned Paloma test suite."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from statistics import fmean
import sys
import time

import torch
from transformers import AutoTokenizer

from flock.eval import hub
from flock.eval.model import ScoringModel
from flock.eval.provenance import Provenance
from flock.eval.results import Results, SuiteResult
from flock.eval.run import score_task
from flock.eval.tasks.paloma import PALOMA
from flock.eval.tasks.suite import pins

from scripts.sync_dclm_flock import ENTITY, PROJECT
from scripts.train_dclm_150m import BOS, DCLM_URI, DATASET_VERSION, EOS, build_model
from upwash.config import ModelConfig


def score(checkpoint, tokenizer_dir, out, *, limit, batch_tokens, allow_training_checkpoint=False):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {out}")
    if root := hub.root():
        hub.check_staged(root, pins(PALOMA.tasks))

    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    run_config = saved["config"]
    if limit is None and not allow_training_checkpoint and (
        saved["step"] != run_config["planned_steps"]
        or saved["tokens"] != run_config["planned_tokens"]
        or run_config["dclm"] != [DCLM_URI, DATASET_VERSION]
    ):
        raise ValueError("Full Paloma evaluation expects the verified final DCLM checkpoint")
    if allow_training_checkpoint and (
        saved["step"] <= 0 or saved["step"] > run_config["planned_steps"]
        or saved["tokens"] != saved["step"] * run_config["tokens_per_step"]
    ):
        raise ValueError("Checkpoint progress does not match the training configuration")
    config = ModelConfig(**saved["config"]["model"])
    model = build_model(config)
    model.load_state_dict(saved["model"])
    model.to(device="cuda", dtype=torch.float32).eval()
    model.set_attn_implementation("sdpa")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (BOS, EOS):
        raise ValueError("Tokenizer does not match the training checkpoint")
    scorer = ScoringModel(model, tokenizer, name=f"upwash:{digest}", digest=digest,
                          batch_tokens=batch_tokens)
    provenance = Provenance.collect(
        model=scorer.name, model_digest=scorer.digest, command=shlex.join(sys.argv),
        device=scorer.device, dtype=scorer.dtype, batch_tokens=scorer.batch_tokens,
        max_length=scorer.max_length, add_bos=scorer.add_bos, attention=scorer.attention,
        sdpa_backend=scorer.sdpa_backend,
    )
    step, tokens = saved["step"], saved["tokens"]
    del saved

    results = []
    tasks = PALOMA.tasks
    for task in tasks:
        result = score_task(scorer, task, limit, out)
        print(f"{result.task} docs={result.num_docs} bpb={result.metrics['bits_per_byte']:.8f}", flush=True)
        results.append(result)

    metrics = {task.task: task.metrics for task in results}
    hashes = {task.task: task.task_hash for task in results}
    suites = []
    for suite in PALOMA.suites:
        members = [member.name for member in suite.members]
        definition = {"suite": suite.name, "metric": suite.metric, "scored": suite.scored,
                      "members": [[name, hashes[name]] for name in members]}
        hashes[suite.name] = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        suites.append(SuiteResult(
            suite.name, hashes[suite.name], members, suite.metric,
            suite.score(metrics) if suite.scored else None,
        ))

    Results(provenance, results, suites).write(out)
    per_source = {suite.suite: suite.score for suite in suites if suite.score is not None}
    summary = {
        "checkpoint_sha256": digest, "step": step, "total_tokens": tokens,
        "limit_per_task": limit, "paloma_source_bpb": per_source,
        "paloma_macro_bpb": fmean(per_source.values()),
    }
    (out / "paloma-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def publish_oj(out, run_id, summary):
    import wandb

    from flock.cli.api import Api
    from flock.cli.credentials import resolve

    if summary["limit_per_task"] is not None or len(summary["paloma_source_bpb"]) != 9:
        raise ValueError("The Lab 7 summary requires all nine Paloma sources without a document limit")
    values = {
        "paloma_ppl": summary["paloma_macro_bpb"],
        "total_tokens": summary["total_tokens"],
        # This baseline scans disjoint source-token positions once.
        "oj_total_tokens": summary["total_tokens"],
    }
    run = wandb.init(
        entity=ENTITY, project=PROJECT, id=run_id, resume="must", dir=out,
        settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False, console="off"),
    )
    if run is None or run.offline or not run.resumed:
        raise RuntimeError("Flock did not resume the existing training run")
    with run:
        run.summary.update({**values, "paloma_macro_bpb": values["paloma_ppl"]})

    credentials = resolve()
    if credentials is None:
        raise RuntimeError("A Flock CLI sign-in is required to verify the published summary")
    api = Api(credentials.host, credentials.token)
    for _ in range(6):
        response = api.get(f"projects/{ENTITY}/{PROJECT}/runs/{run_id}")
        response.raise_for_status()
        remote = response.json()["summary"]
        if all(remote.get(key) == value for key, value in values.items()):
            print(f"FLOCK verified OJ summary {values}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"Flock did not return the published OJ summary for {run_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--flock-run-id")
    args = parser.parse_args()
    if args.flock_run_id and args.limit is not None:
        parser.error("Publishing the Lab 7 summary requires the full suite")
    summary = score(args.checkpoint, args.tokenizer_dir, args.out,
                    limit=args.limit, batch_tokens=args.batch_tokens)
    if args.flock_run_id:
        publish_oj(args.out, args.flock_run_id, summary)


if __name__ == "__main__":
    main()
