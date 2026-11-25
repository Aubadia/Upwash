"""Prepare, score, and rank DataDecide recipes on Upwash's Paloma validation."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean, stdev
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile

STEP = 69369
SEEDS = ("default", "large-aux-2", "large-aux-3")
RECIPES = {
    "DCLM-Baseline (QC 20%)": "dclm-baseline-qc-20p",
    "Falcon+CC (QC Orig 10%)": "falcon-and-cc-qc-orig-10p",
    "DCLM-Baseline (QC 7%, FW2)": "dclm-baseline-qc-7p-fw2",
    "DCLM-Baseline (QC 10%)": "dclm-baseline-qc-10p",
    "Falcon+CC (QC 10%)": "falcon-and-cc-qc-10p",
    "DCLM-Baseline (QC FW 10%)": "dclm-baseline-qc-fw-10p",
    "Falcon+CC (QC 20%)": "falcon-and-cc-qc-20p",
    "DCLM-Baseline": "dclm-baseline",
    "FineWeb-Pro": "fineweb-pro",
    "FineWeb-Edu": "fineweb-edu",
}
XLSX = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CONTEXT = 2048
DOCUMENTS = 128


def xlsx_rows(archive, title):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(sheet for sheet in workbook.findall(".//x:sheet", XLSX) if sheet.get("name") == title)
    relationship = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    path = next(item.attrib["Target"] for item in relationships if item.get("Id") == relationship)
    path = path.lstrip("/") if path.startswith("/") else f"xl/{path}"
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.findall(".//x:t", XLSX)) for item in strings]

    document = ET.fromstring(archive.read(path))
    for row in document.findall(".//x:sheetData/x:row", XLSX):
        cells = {}
        for cell in row.findall("x:c", XLSX):
            column = cell.attrib["r"].rstrip("0123456789")
            value = cell.findtext("x:v", namespaces=XLSX)
            if cell.get("t") == "s":
                value = shared[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//x:t", XLSX))
            if value is not None:
                cells[column] = value
        yield cells


def top_recipes(workbook):
    with zipfile.ZipFile(workbook) as archive:
        rows = list(xlsx_rows(archive, "Macro ranks"))
        coverage = list(xlsx_rows(archive, "Method and coverage"))
    heading = next(row for row in rows if "Training recipe" in row.values() and "1B" in row.values())
    name_col = next(column for column, value in heading.items() if value == "Training recipe")
    rank_col = next(column for column, value in heading.items() if value == "1B")
    checkpoint = next(row for row in coverage if row.get("A") == "1B")
    if int(checkpoint["B"]) != STEP or int(checkpoint["F"]) != len(SEEDS):
        raise ValueError("Workbook 1B checkpoint coverage differs from the expected release")
    ranked = sorted(
        (int(row[rank_col]), row[name_col]) for row in rows
        if row.get(rank_col, "").isdigit() and 1 <= int(row[rank_col]) <= 10
    )
    if [rank for rank, _ in ranked] != list(range(1, 11)) or {name for _, name in ranked} != RECIPES.keys():
        raise ValueError("Expected ten distinct mapped recipes at workbook ranks 1–10")
    return ranked


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text())


def prepare(workbook, run_dir):
    from huggingface_hub import HfApi

    from scripts.train_dclm_150m import (
        DATASET_VERSION, PALOMA_URI, open_datasets, paloma_texts, validation_texts,
    )

    api = HfApi()
    recipes = []
    for rank, name in top_recipes(workbook):
        slug = RECIPES[name]
        repo = f"allenai/DataDecide-{slug}-1B"
        refs = api.list_repo_refs(repo)
        revisions = {ref.name: ref.target_commit for ref in [*refs.branches, *refs.tags]}
        seeds = {seed: revisions[f"step{STEP}-seed-{seed}"] for seed in SEEDS}
        if any(not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in seeds.values()) or len(set(seeds.values())) != 3:
            raise ValueError(f"{repo} does not have three distinct pinned checkpoint revisions")
        recipes.append({"macro_rank": rank, "recipe": name, "slug": slug, "repo": repo, "revisions": seeds})

    (paloma,) = open_datasets((PALOMA_URI, DATASET_VERSION))
    texts = validation_texts(paloma_texts(paloma))
    nbytes = sum(len(text.encode("utf-8")) for text in texts)
    if len(texts) != DOCUMENTS or nbytes != 733644:
        raise ValueError(f"Unexpected Paloma v1 validation subset: {len(texts)} documents, {nbytes} bytes")
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / "validation.json"
    atomic_json(snapshot, texts)
    atomic_json(run_dir / "manifest.json", {
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "step": STEP,
        "validation": {
            "uri": PALOMA_URI, "version": DATASET_VERSION, "documents": len(texts),
            "bytes": nbytes, "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        },
        "recipes": recipes,
    })


def score(run_dir, macro_rank):
    import gc

    import torch
    from hf_olmo import OLMoForCausalLM
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    from scripts.paloma_validation import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    manifest = read_json(run_dir / "manifest.json")
    recipe = next(item for item in manifest["recipes"] if item["macro_rank"] == macro_rank)
    snapshot = (run_dir / "validation.json").read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()
    texts = json.loads(snapshot)
    validation = manifest["validation"]
    if digest != validation["sha256"] or len(texts) != DOCUMENTS:
        raise ValueError("Validation snapshot does not match the manifest")

    out = run_dir / recipe["slug"]
    out.mkdir(exist_ok=True)
    for seed, revision in recipe["revisions"].items():
        model_dir = snapshot_download(
            recipe["repo"], revision=revision,
            allow_patterns=["config.json", "generation_config.json", "tokenizer.json",
                            "tokenizer_config.json", "special_tokens_map.json",
                            "*.safetensors", "model.safetensors.index.json"],
        )
        model, loading = OLMoForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, attn_implementation="sdpa",
            use_safetensors=True, local_files_only=True, output_loading_info=True,
        )
        if any(loading[key] for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")):
            raise RuntimeError(f"Checkpoint load differed from OLMo architecture for {seed}: {loading}")
        if model.config.max_sequence_length != CONTEXT:
            raise ValueError(f"Unexpected {seed} context length: {model.config.max_sequence_length}")
        model.config.use_cache = False
        tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        prefix = tokenizer.token_to_id("<|endoftext|>")
        if prefix is None or prefix != model.config.eos_token_id:
            raise ValueError(f"Unexpected {seed} EOS token ID")
        result = evaluate(model.to("cuda"), tokenizer, texts, "cuda",
                          prefix_token_id=prefix, seq_len=CONTEXT)
        record = {
            "repo": recipe["repo"], "revision": revision, "seed": seed,
            "step": manifest["step"], "validation_sha256": digest, "documents": len(texts),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "torch": str(torch.__version__), "context": CONTEXT, "prefix_token_id": prefix,
            **result,
        }
        atomic_json(out / f"{seed}.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def ranking_rows(run_dir, manifest, jobs):
    errors = []
    rows = []
    for recipe in manifest["recipes"]:
        name = recipe["recipe"]
        job = jobs[str(recipe["macro_rank"])]
        scores = {}
        for seed, revision in recipe["revisions"].items():
            path = run_dir / recipe["slug"] / f"{seed}.json"
            try:
                result = read_json(path)
                expected = {
                    "repo": recipe["repo"], "revision": revision, "seed": seed,
                    "step": manifest["step"], "slurm_job_id": str(job),
                    "validation_sha256": manifest["validation"]["sha256"], "documents": DOCUMENTS,
                    "eval/bytes": manifest["validation"]["bytes"],
                }
                if any(result.get(key) != value for key, value in expected.items()):
                    raise ValueError("checkpoint, Slurm ID or validation snapshot mismatch")
                if (not math.isfinite(result["eval/bpb"]) or not math.isfinite(result["eval/loss"])
                        or result["eval/tokens"] <= 0):
                    raise ValueError("non-finite score or missing token count")
                scores[seed] = result["eval/bpb"]
            except (OSError, KeyError, ValueError, TypeError) as error:
                errors.append(f"{name} / {seed} / Slurm {job}: {error}")
        if len(scores) == len(SEEDS):
            rows.append({
                "macro_rank": recipe["macro_rank"], "recipe": name, **scores,
                "mean_bpb": fmean(scores.values()), "seed_sd": stdev(scores.values()),
            })
    if errors:
        raise ValueError("\n" + "\n".join(errors))
    if len(rows) != 10:
        raise ValueError(f"Expected 10 complete recipe results, got {len(rows)}")
    rows.sort(key=lambda row: (row["mean_bpb"], row["macro_rank"]))
    previous = None
    for position, row in enumerate(rows, 1):
        if row["mean_bpb"] != previous:
            rank = position
        row["paloma_rank"] = rank
        previous = row["mean_bpb"]
    return rows


def rank(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    jobs = read_json(run_dir / "jobs.json")["evaluation"]
    rows = ranking_rows(run_dir, manifest, jobs)
    for _ in range(6):
        accounting = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()), "-o", "JobIDRaw,State"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        states = dict(line.split("|", 1) for line in accounting.stdout.splitlines() if line.strip())
        pending = {job: states.get(job, "missing") for job in jobs.values()
                   if states.get(job) != "COMPLETED"}
        if not pending:
            break
        time.sleep(5)
    else:
        raise RuntimeError(f"Evaluation jobs without a successful Slurm exit: {pending}")
    fields = ("paloma_rank", "macro_rank", "recipe", *SEEDS, "mean_bpb", "seed_sd")
    with (run_dir / "ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    header = ("Paloma rank", "MacroRank", "Recipe", *SEEDS, "Mean BPB", "Sample SD")
    lines = [
        "# DataDecide 1B Paloma validation ranking", "",
        "Upwash Paloma v1 validation subset (128 documents). Three seeds per recipe. Lower BPB ranks higher.", "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [f"{row[key]:.6f}" if key in (*SEEDS, "mean_bpb", "seed_sd") else str(row[key])
                  for key in fields]
        lines.append("| " + " | ".join(values) + " |")
    text = "\n".join(lines) + "\n"
    (run_dir / "ranking.md").write_text(text)
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "score", "rank"):
        command = commands.add_parser(name)
        command.add_argument("run_dir", type=Path)
        if name == "prepare":
            command.add_argument("workbook", type=Path)
        if name == "score":
            command.add_argument("macro_rank", type=int, choices=range(1, 11))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.workbook, args.run_dir)
    elif args.command == "score":
        score(args.run_dir, args.macro_rank)
    else:
        rank(args.run_dir)


if __name__ == "__main__":
    main()
