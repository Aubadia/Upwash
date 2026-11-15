"""Overnight 150M sweep driver (S1 -> S2 -> S3), run with nohup on the login node.

Rules agreed with the user on 2026-11-14:
- Start only after the pool job ends and the pool passes verification.
- Submit waves of at most 10 jobs; the next wave starts only after every job
  of the current wave has ended.
- Best = lowest final eval/loss (from the wandb summary at the end of the log).
- Stop if the S1 best lr is on the grid edge, after any code error, and after S3.
- A node failure (GPU enumeration, NCCL bootstrap, NODE_FAIL) is resubmitted
  once with that node excluded; a second failure counts as a code error.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/work/u7725908/Upwash")
HERE = Path("/work/u7725908/upwash/sweep")
POOL = Path("/work/u7725908/fineweb-edu-llama3-upwash")
POOL_JOB = "773200"
BASE = REPO / "configs" / "sweep-150m-base.json"
CONFIG_DIR = REPO / "configs" / "sweep150m"
STATE = HERE / "state.json"
RESULTS = HERE / "RESULTS.md"
EXCLUDE = ["25a-hgpn001", "25a-hgpn003", "25a-hgpn020", "25a-hgpn039", "25a-hgpn056",
           "25a-hgpn062", "25a-hgpn069", "25a-hgpn071", "25a-hgpn083", "25a-hgpn173"]
WAVE_LIMIT = 10
POLL = 60
TIME_LIMIT = "02:00:00"

LRS = [0.002, 0.004, 0.008, 0.016, 0.032]
MUON_BETAS = [0.9, 0.95, 0.98, 0.99]
BETA2S = [0.95, 0.98, 0.99]
MULTIPLIERS = [1.5, 1.6, 1.7]


def log(message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with open(HERE / "driver.log", "a") as f:
        f.write(line + "\n")


def results(text):
    with open(RESULTS, "a") as f:
        f.write(text + "\n")


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"runs": {}, "excluded": list(EXCLUDE)}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=1))


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, check=False)


def job_active(job_id):
    return bool(sh("squeue", "-h", "-j", str(job_id)).stdout.strip())


def job_info(job_id):
    out = sh("sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "State,NodeList,ExitCode").stdout.strip()
    state, node, code = (out.splitlines()[0].split("|") + ["", "", ""])[:3] if out else ("UNKNOWN", "", "")
    return state.split()[0], node, code


# ---------------------------------------------------------------- pool check
def verify_pool():
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from upwash.builder import build_dataset
    from upwash.config import load_run_config

    problems = []
    manifest_path = POOL / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    shards = POOL / "shards"
    bins = sorted(shards.glob("*.bin"))
    jsons = sorted(shards.glob("*.json"))
    tmps = sorted(shards.glob("*.tmp"))
    if manifest["total_tokens"] < 205_000_000_000:
        problems.append(f"total_tokens {manifest['total_tokens']:,} < 205e9")
    if len(bins) != len(manifest["shards"]) or len(jsons) != len(bins):
        problems.append(f"{len(bins)} .bin / {len(jsons)} .json vs {len(manifest['shards'])} manifest shards")
    if tmps:
        problems.append(f"{len(tmps)} leftover .tmp files")
    total = 0
    for meta in manifest["shards"]:
        path = shards / f"{meta['index']:05d}.bin"
        size = path.stat().st_size if path.exists() else -1
        if size != 4 * meta["num_tokens"]:
            problems.append(f"{path.name}: {size} bytes != 4 x {meta['num_tokens']}")
        total += meta["num_tokens"]
    if total != manifest["total_tokens"]:
        problems.append(f"shard token sum {total:,} != manifest {manifest['total_tokens']:,}")
    rng = np.random.default_rng(0)
    for path in rng.choice(bins, size=min(5, len(bins)), replace=False):
        tokens = np.memmap(path, dtype="<u4", mode="r")
        bos, eos = int((tokens == 128000).sum()), int((tokens == 128001).sum())
        if tokens[0] != 128000 or tokens[-1] != 128001 or bos != eos or int(tokens.max()) > 128001:
            problems.append(f"{Path(path).name}: first={tokens[0]} last={tokens[-1]} bos={bos} eos={eos}")
    config = load_run_config(BASE)
    dataset = build_dataset(config.train_data)
    expected = config.train_data.num_tokens // config.train_data.seq_len
    if len(dataset) != expected:
        problems.append(f"dataset has {len(dataset)} samples, expected {expected}")
    x, y = dataset[0]
    if x.shape[0] != 4096 or not bool((x[1:] == y[:-1]).all()):
        problems.append("dataset sample shape/shift wrong")
    results(
        f"## Pool verification ({datetime.now():%Y-%m-%d %H:%M})\n\n"
        f"- total_tokens {manifest['total_tokens']:,}, docs {manifest['num_docs']:,}, shards {len(bins)}, "
        f"dev ids excluded {manifest['dev_set']['excluded']}, seed {manifest['seed']}\n"
        f"- problems: {problems or 'none'}\n"
    )
    return problems


# ---------------------------------------------------------------- runs
def make_config(name, group, seed=None, **optimizer):
    config = json.loads(BASE.read_text())
    unknown = set(optimizer) - set(config["training"]["optimizer"])
    if unknown:
        raise KeyError(f"unknown optimizer fields {unknown}")
    config["training"]["optimizer"].update(optimizer)
    config["logging"].update(name=name, group=group)
    if seed is not None:
        config["seed"] = seed
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def submit(state, name, path):
    env = dict(os.environ, CONFIG=str(path.relative_to(REPO)), RUN=name)
    out = subprocess.run(
        ["sbatch", "-J", name, "--partition=8gpus", f"--time={TIME_LIMIT}",
         f"--exclude={','.join(state['excluded'])}", "scripts/train.sbatch"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    match = re.search(r"Submitted batch job (\d+)", out.stdout)
    if not match:
        raise RuntimeError(f"sbatch failed for {name}: {out.stdout} {out.stderr}")
    run = state["runs"].setdefault(name, {"config": str(path), "attempts": []})
    run["attempts"].append(match.group(1))
    save_state(state)
    log(f"submitted {name} as job {match.group(1)}")
    return match.group(1)


NODE_PATTERNS = ("nvidia-smi cannot enumerate", "GPUs visible on", "NCCL", "CUDA error: no CUDA-capable",
                 "torch cannot use")


def classify(job_id, name):
    state, node, code = job_info(job_id)
    log_path = REPO / "logs" / f"{name}-{job_id}.out"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    summary = {}
    for key in ("eval/loss", "eval/perplexity", "train/loss", "train/tokens_per_second", "train/total_tokens_seen"):
        found = re.findall(rf"wandb:\s+{re.escape(key)} ([0-9.eE+-]+|nan|inf)\s*$", text, flags=re.M)
        if found:
            summary[key] = float(found[-1])
    if summary.get("eval/loss", 0.0) != summary.get("eval/loss", 0.0):
        # NaN divergence is a valid (worst) result, not a code error.
        summary["eval/loss"] = float("inf")
        summary["diverged"] = True
    ok = state == "COMPLETED" and "done:" in text and "eval/loss" in summary
    if ok:
        return "ok", node, summary
    if state in ("NODE_FAIL",) or any(p in text for p in NODE_PATTERNS):
        return "node", node, summary
    return "code", node, {"state": state, "exit": code, "tail": text[-1500:]}


def run_wave(state, wave):
    """wave: list of (name, path). Returns {name: summary} or None on a code error."""
    pending = {}
    for name, path in wave:
        run = state["runs"].get(name, {})
        if run.get("result"):
            continue
        attempts = run.get("attempts", [])
        if attempts and job_active(attempts[-1]):
            pending[name] = attempts[-1]  # driver restarted while the job was queued or running
            log(f"resuming {name} job {attempts[-1]}")
            continue
        if attempts:  # ended while the driver was down: classify it below
            pending[name] = attempts[-1]
            continue
        pending[name] = submit(state, name, path)
    retried = set()
    while pending:
        time.sleep(POLL)
        for name, job_id in list(pending.items()):
            if job_active(job_id):
                continue
            kind, node, info = classify(job_id, name)
            if kind == "ok":
                state["runs"][name]["result"] = info
                save_state(state)
                log(f"{name} ok: eval/loss {info['eval/loss']:.4f}, tok/s {info.get('train/tokens_per_second', 0):,.0f}")
                del pending[name]
            elif kind == "node" and name not in retried:
                log(f"{name} job {job_id} node failure on {node}; excluding it and resubmitting")
                if node and node not in state["excluded"]:
                    state["excluded"].append(node)
                retried.add(name)
                pending[name] = submit(state, name, Path(state["runs"][name]["config"]))
            else:
                log(f"{name} job {job_id} failed ({kind}): {info}")
                results(f"\n**Stopped:** {name} (job {job_id}) failed: `{info.get('state')}` exit `{info.get('exit')}`.\n\n```\n{info.get('tail', '')}\n```\n")
                # Let the rest of this wave finish, then stop.
                del pending[name]
                state["stopped"] = f"{name} failed"
                save_state(state)
    if state.get("stopped"):
        return None
    return {name: state["runs"][name]["result"] for name, _ in wave}


def table(title, rows):
    lines = [f"\n## {title}\n", "| run | eval/loss | eval PPL | train/loss | tok/s |", "|---|---|---|---|---|"]
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["eval/loss"]):
        lines.append(f"| {name} | {r['eval/loss']:.4f} | {r.get('eval/perplexity', float('nan')):.2f} | "
                     f"{r.get('train/loss', float('nan')):.4f} | {r.get('train/tokens_per_second', 0):,.0f} |")
    results("\n".join(lines))


def fmt(x):
    return f"{x:g}"


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log("driver started")

    if not state.get("pool_ok"):
        while job_active(POOL_JOB):
            time.sleep(POLL)
        pool_state, _, _ = job_info(POOL_JOB)
        log(f"pool job {POOL_JOB} ended: {pool_state}")
        if pool_state != "COMPLETED":
            results(f"\n**Stopped:** pool job {POOL_JOB} ended {pool_state}; no sweep submitted.\n")
            return
        problems = verify_pool()
        if problems:
            log(f"pool verification failed: {problems}")
            results("\n**Stopped:** pool verification failed; no sweep submitted.\n")
            return
        state["pool_ok"] = True
        save_state(state)
        log("pool verified")

    # S1: lr x Muon beta, two waves of 10.
    s1 = {}
    for betas in (MUON_BETAS[:2], MUON_BETAS[2:]):
        wave = []
        for mb in betas:
            for lr in LRS:
                name = f"s1-lr{fmt(lr)}-mb{fmt(mb)}"
                wave.append((name, make_config(name, "sweep150m-s1", lr=lr, muon_beta=mb)))
        out = run_wave(state, wave)
        if out is None:
            return
        s1.update(out)
    table("S1: lr x Muon beta (beta1 0.85, beta2 0.95, multiplier 1.5)", s1)

    def s1_best():
        name = min(s1, key=lambda n: s1[n]["eval/loss"])
        cfg = json.loads(Path(state["runs"][name]["config"]).read_text())["training"]["optimizer"]
        return name, cfg["lr"], cfg["muon_beta"]

    best_name, lr, mb = s1_best()
    results(f"\nS1 best: **{best_name}** (lr {lr}, Muon beta {mb}).")
    grid = sorted(LRS)
    if lr in (grid[0], grid[-1]):
        # One extension wave: two more points beyond the edge for every Muon beta (8 runs).
        extra = [grid[-1] * 2, grid[-1] * 4] if lr == grid[-1] else [grid[0] / 2, grid[0] / 4]
        results(f"\nS1 best lr {lr} is on the grid edge; extending with lr {extra} for every Muon beta.")
        log(f"S1 edge at lr {lr}; extending with {extra}")
        wave = []
        for m in MUON_BETAS:
            for x in extra:
                name = f"s1-lr{fmt(x)}-mb{fmt(m)}"
                wave.append((name, make_config(name, "sweep150m-s1", lr=x, muon_beta=m)))
        out = run_wave(state, wave)
        if out is None:
            return
        s1.update(out)
        table("S1 with edge extension", s1)
        grid = sorted(grid + extra)
        best_name, lr, mb = s1_best()
        results(f"\nS1 best after extension: **{best_name}** (lr {lr}, Muon beta {mb}).")
        if lr in (grid[0], grid[-1]):
            results(f"\n**Stopped:** S1 best lr {lr} is still on the extended grid edge {grid}; review before S2.\n")
            log("stopped: S1 best lr still on grid edge after extension")
            return

    # S2: beta1 0.9, plus a seed replica of the S1 best for a noise estimate.
    seed_name = f"s2-noise-lr{fmt(lr)}-mb{fmt(mb)}-seed1"
    b1_name = f"s2-lr{fmt(lr)}-mb{fmt(mb)}-b1_0.9"
    wave = [(b1_name, make_config(b1_name, "sweep150m-s2", lr=lr, muon_beta=mb, adam_beta1=0.9)),
            (seed_name, make_config(seed_name, "sweep150m-s2", lr=lr, muon_beta=mb, seed=1))]
    out = run_wave(state, wave)
    if out is None:
        return
    noise = abs(out[seed_name]["eval/loss"] - s1[best_name]["eval/loss"])
    table("S2: AdamW beta1 and seed replica", {best_name: s1[best_name], **out})
    beta1 = 0.9 if out[b1_name]["eval/loss"] < s1[best_name]["eval/loss"] else 0.85
    results(f"\nSeed-to-seed |delta eval/loss| = {noise:.4f}. beta1 chosen: **{beta1}** "
            f"(0.9: {out[b1_name]['eval/loss']:.4f} vs 0.85: {s1[best_name]['eval/loss']:.4f}).")

    # S3: AdamW beta2 x LR multiplier (the (0.95, 1.5) point is the current best).
    reference = best_name if beta1 == 0.85 else b1_name
    reference_result = s1[best_name] if beta1 == 0.85 else out[b1_name]
    wave = []
    for b2 in BETA2S:
        for m in MULTIPLIERS:
            if (b2, m) == (0.95, 1.5):
                continue
            name = f"s3-lr{fmt(lr)}-mb{fmt(mb)}-b1_{fmt(beta1)}-b2_{fmt(b2)}-m{fmt(m)}"
            wave.append((name, make_config(name, "sweep150m-s3", lr=lr, muon_beta=mb, adam_beta1=beta1,
                                           adam_beta2=b2, adam_lr_multiplier=m)))
    out = run_wave(state, wave)
    if out is None:
        return
    table("S3: AdamW beta2 x LR multiplier", {reference: reference_result, **out})
    best3 = min({reference: reference_result, **out}.items(), key=lambda kv: kv[1]["eval/loss"])[0]
    results(f"\nS3 best: **{best3}**. Stopped after S3 as agreed; warmup/decay/shape/weight decay await review.\n")
    log("finished S1-S3")


if __name__ == "__main__":
    main()
