"""Backfill and follow a running DCLM job's JSONL metrics in Flock."""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import wandb

ENTITY = "cerulean-labs"
PROJECT = "lab7-data-curation"
ACTIVE_STATES = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "REQUEUED", "SUSPENDED"}


def job_state(job_id):
    try:
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "State"],
            text=True, check=True, capture_output=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return next((line.split()[0].rstrip("+|") for line in result.stdout.splitlines() if line.strip()), "")


def drain_metrics(source, run, next_step):
    while True:
        offset = source.tell()
        line = source.readline()
        if not line.endswith("\n"):
            # Leave an incomplete final record for the next read.
            source.seek(offset)
            return next_step
        metrics = json.loads(line)
        step = metrics.pop("step")
        if step >= next_step:
            run.log({"train/global_step": step, **metrics}, step=step)
            next_step = step + 1


def final_summary(run_dir, config, observed):
    tokens = observed["train/total_tokens_seen"]
    step = observed["step"]
    if "recipe" not in config:
        if (not (run_dir / "final.pt").is_file()
                or step != config["planned_steps"] or tokens != config["planned_tokens"]):
            raise RuntimeError("Training exited without the expected final checkpoint and metrics")
        return {"total_tokens": tokens}

    completion = json.loads((run_dir / "completion.json").read_text())
    final = run_dir / "final.pt"
    metrics = completion["metrics"]
    official = metrics["paloma/macro_bpb"]
    local = metrics["eval/bpb"]
    threshold = config["paloma_threshold"]
    qualified = local <= threshold and official <= threshold
    early = qualified and step < config["planned_steps"]
    reason = "paloma_threshold" if early else "token_limit"
    if (not final.is_file() or Path(completion["checkpoint"]) != final
            or not final.samefile(run_dir / "latest.pt")
            or step != completion["step"] or not 0 < step <= config["planned_steps"]
            or (not early and step != config["planned_steps"])
            or tokens != step * config["tokens_per_step"] or tokens != completion["tokens"]
            or not math.isfinite(official) or not math.isfinite(local)
            or observed["paloma/macro_bpb"] != official or observed["eval/bpb"] != local
            or completion["qualified"] != qualified or observed["paloma/qualified"] != qualified
            or completion["stop_reason"] != reason):
        raise RuntimeError("Recipe completion disagrees with the checkpoint or Paloma metrics")
    # The recipe sampler traverses its source positions once, so no epoch
    # repetition discount applies to the tokens that reached the optimizer.
    return {
        "oj_total_tokens": tokens, "total_tokens": tokens, "paloma_ppl": official,
        "paloma_bpb": official, "upwash_paloma_bpb": local,
        "paloma_qualified": qualified, "stop_reason": reason,
    }


def sync(run_dir, job_id, flock_run_id=None, after_step=-1):
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    resume_marker = run_dir / f".resume-{job_id}.ready"
    while not (config_path.is_file() and metrics_path.is_file()
               and (not flock_run_id or resume_marker.is_file())):
        state = job_state(job_id)
        if state and state not in ACTIVE_STATES:
            raise RuntimeError(f"Slurm job {job_id} ended in {state} before writing metrics")
        time.sleep(15)
    config = json.loads(config_path.read_text())
    sync_dir = run_dir / "flock"
    sync_dir.mkdir(exist_ok=True)
    run = wandb.init(
        entity=ENTITY, project=PROJECT, id=flock_run_id or f"d{job_id}", name=run_dir.name,
        config={**config, "slurm_job_id": int(flock_run_id[1:]) if flock_run_id else job_id},
        tags=["upwash", "150m", *(["datadecide", config["recipe"]] if "recipe" in config
                               else ["dclm", "baseline"])],
        dir=sync_dir, resume="must" if flock_run_id else "allow",
        settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False, console="off"),
    )
    if run is None or run.offline:
        raise RuntimeError("Flock run did not start online")
    print(f"FLOCK run={ENTITY}/{PROJECT}/{run.id} resumed={run.resumed} next_step={run.step}", flush=True)
    if flock_run_id:
        run.config.update({"active_slurm_job_id": job_id, "paloma_policy": "full_horizon"},
                          allow_val_change=True)
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")
    exit_code = 1
    try:
        next_step = max(run.step, after_step + 1)
        with metrics_path.open() as source:
            while True:
                before = next_step
                next_step = drain_metrics(source, run, next_step)
                if next_step != before:
                    print(f"SYNC through_step={next_step - 1}", flush=True)

                state = job_state(job_id)
                if not state or state in ACTIVE_STATES:
                    time.sleep(15)
                    continue

                # Slurm can report terminal status while the final log is in flight.
                next_step = drain_metrics(source, run, next_step)
                if state == "COMPLETED":
                    observed = json.loads(metrics_path.read_text().splitlines()[-1])
                    if next_step != observed["step"] + 1:
                        raise RuntimeError("Final metrics were not synchronized")
                    run.summary.update(final_summary(run_dir, config, observed))
                    exit_code = 0
                print(f"END training_state={state} through_step={next_step - 1}", flush=True)
                break
    finally:
        run.finish(exit_code=exit_code)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("slurm_job_id", type=int)
    parser.add_argument("--flock-run-id", help="Resume the original Flock run under a replacement Slurm job.")
    parser.add_argument("--after-step", type=int, default=-1, help="Skip steps already verified in Flock.")
    args = parser.parse_args()
    if args.flock_run_id and not (args.flock_run_id.startswith("d") and args.flock_run_id[1:].isdigit()):
        parser.error("--flock-run-id must be the original Slurm-backed Flock ID")
    raise SystemExit(sync(args.run_dir.resolve(), args.slurm_job_id, args.flock_run_id, args.after_step))


if __name__ == "__main__":
    main()
