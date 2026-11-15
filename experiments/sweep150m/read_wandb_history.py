"""Read train/eval curves from local wandb run files (the Flock server rejects wandb.Api).

Usage: .venv/bin/python experiments/sweep150m/read_wandb_history.py
Prints, per run: first and min train loss, the first spike (> min + 1), and every eval/loss.
"""
import sys, glob, json, struct
from wandb.proto import wandb_internal_pb2 as pb
BLOCK = 32768
def records(path):
    buf = open(path, "rb").read()
    pos, pending = 7, b""
    while pos + 7 <= len(buf):
        left = BLOCK - pos % BLOCK
        if left < 7: pos += left; continue
        _, n, t = struct.unpack("<IHB", buf[pos:pos+7]); pos += 7
        if n == 0 and t == 0: break
        chunk = buf[pos:pos+n]; pos += n
        if t == 1: yield chunk
        elif t == 2: pending = chunk
        elif t == 3: pending += chunk
        elif t == 4: yield pending + chunk; pending = b""
def history(path):
    rows, name = [], None
    for data in records(path):
        rec = pb.Record(); rec.ParseFromString(data)
        t = rec.WhichOneof("record_type")
        if t == "run": name = rec.run.display_name
        if t == "history":
            rows.append({(i.key or "/".join(i.nested_key)): json.loads(i.value_json) for i in rec.history.item})
    return name, rows
if __name__ == "__main__":
    for d in sorted(glob.glob("/work/u7725908/upwash/wandb/wandb/run-20261114_0[67]*")):
        name, rows = history(glob.glob(d + "/*.wandb")[0])
        ev = [round(r["eval/loss"], 2) for r in rows if "eval/loss" in r]
        tl = [(r.get("_step"), r["train/loss"]) for r in rows if "train/loss" in r]
        mn, spike = 99, None
        for s, l in tl:
            if l != l: spike = spike or ("nan@", s); break
            if spike is None and l > mn + 1.0: spike = (round(l, 2), "@", s, "min", round(mn, 2))
            mn = min(mn, l)
        print(f"{name:20s} first {tl[0][1]:.2f} min {mn:.2f} n={len(tl)} spike {spike} eval {ev}")
