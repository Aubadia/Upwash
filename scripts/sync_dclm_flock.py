"""Backfill and follow a running DCLM job's JSONL metrics in Flock."""

import argparse
import json
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


def sync(run_dir, job_id):
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    while not (config_path.is_file() and metrics_path.is_file()):
        state = job_state(job_id)
        if state and state not in ACTIVE_STATES:
            raise RuntimeError(f"Slurm job {job_id} ended in {state} before writing metrics")
        time.sleep(15)
    config = json.loads(config_path.read_text())
    sync_dir = run_dir / "flock"
    sync_dir.mkdir(exist_ok=True)
    run = wandb.init(
        entity=ENTITY, project=PROJECT, id=f"d{job_id}", name=run_dir.name,
        config={**config, "slurm_job_id": job_id},
        tags=["upwash", "dclm", "150m", "baseline"],
        dir=sync_dir, resume="allow",
        settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False, console="off"),
    )
    if run is None or run.offline:
        raise RuntimeError("Flock run did not start online")
    print(f"FLOCK run={ENTITY}/{PROJECT}/{run.id} resumed={run.resumed} next_step={run.step}", flush=True)
    run.define_metric("train/global_step")
    run.define_metric("*", step_metric="train/global_step")
    exit_code = 1
    try:
        next_step = run.step
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
                    if not (run_dir / "final.pt").is_file() or next_step != config["planned_steps"] + 1:
                        raise RuntimeError("Training exited without the expected final checkpoint and metrics")
                    observed = json.loads(metrics_path.read_text().splitlines()[-1])
                    if observed["train/total_tokens_seen"] != config["planned_tokens"]:
                        raise RuntimeError("Final logged token count disagrees with the run configuration")
                    run.summary["total_tokens"] = observed["train/total_tokens_seen"]
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
    args = parser.parse_args()
    raise SystemExit(sync(args.run_dir.resolve(), args.slurm_job_id))


if __name__ == "__main__":
    main()
