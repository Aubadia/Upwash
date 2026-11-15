"""Tokenize a shuffled FineWeb-Edu pool for pretraining.

Source files are taken in a seeded random order (mixing CC dumps) and tokenized
in parallel as [BOS, text, EOS] uint32 shards, excluding the dev-set ids. Shard
k is always the k-th file of that order, so a larger --target-tokens extends the
same pool, and reruns skip finished shards. Document-level mixing happens in the
loader: set DataConfig.shuffle_seed.

    python scripts/prepare_pretrain.py --target-tokens 205e9 --workers 96

Train with DataConfig(root=<output-dir>/shards, pattern="*.bin", format="bin",
seq_len=4096, num_tokens=N, shuffle_seed=0).
"""

import argparse
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np

from upwash.core.data import load_dev_ids, read_parquet_rows, write_token_shard

DATA_REPO = "HuggingFaceFW/fineweb-edu"
DATA_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

DEV_REPO = "cerulean-works/fineweb-edu-dev"
DEV_REVISION = "2484afe000a6fdf442ba2396391a3bec12644903"
DEV_FILENAME = "dev-ids.txt"

TOKENIZER_REPO = "meta-llama/Llama-3.2-1B"
TOKENIZER_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
BOS_ID, EOS_ID = 128000, 128001


def list_source_files(repo=DATA_REPO, revision=DATA_REVISION):
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def shuffled_files(files, seed):
    files = sorted(files)
    order = np.random.default_rng(seed).permutation(len(files))
    return [files[i] for i in order]


def shard_paths(output_dir, index):
    bin_path = Path(output_dir) / "shards" / f"{index:05d}.bin"
    return bin_path, bin_path.with_suffix(".json")


def process_file(index, filename, *, output_dir, download_dir, tokenize, exclude_ids, download):
    """Tokenize one source file into shard `index`; reuse it if already finished."""
    bin_path, json_path = shard_paths(output_dir, index)
    if json_path.exists():
        return json.loads(json_path.read_text())

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    local_dir = Path(download_dir) / f"{index:05d}"
    try:
        path = download(filename, local_dir)
        rows = read_parquet_rows(path, ["id", "text"])
        meta = write_token_shard(
            rows, tokenize=tokenize, eos_id=EOS_ID, exclude_ids=exclude_ids, output_path=bin_path,
        )
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)  # raw parquet is not kept
    meta = {"index": index, "source": filename, **meta}
    json_path.write_text(json.dumps(meta))
    return meta


def prepare(files, *, output_dir, target_tokens, seed, run_file, workers=0):
    """Process files in seeded order until the prefix holds target_tokens; write a manifest."""
    order = shuffled_files(files, seed)
    used, total = [], 0
    tasks = list(enumerate(order))
    if workers:
        pool = multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker,
                                                        initargs=(_worker_settings,))
        results = pool.imap(_run_task, tasks)
    else:
        pool, results = None, (run_file(index, name) for index, name in tasks)
    try:
        for meta in results:
            used.append(meta)
            total += meta["num_tokens"]
            print(f"shard {meta['index']:05d} {meta['num_tokens']:,} tokens; pool {total:,}", flush=True)
            if total >= target_tokens:
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    if total < target_tokens:
        raise ValueError(f"all {len(order)} files hold only {total:,} tokens < {target_tokens:,}")

    # Shards past the prefix (finished by workers running ahead) are not part of
    # the pool; remove them so the training glob matches exactly the manifest.
    keep = {shard_paths(output_dir, m["index"])[0].name for m in used}
    for path in (Path(output_dir) / "shards").iterdir():
        if path.with_suffix(".bin").name not in keep:
            path.unlink()

    manifest = {
        "source": {"repo": DATA_REPO, "revision": DATA_REVISION},
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION},
        "format": "[BOS, text, EOS] uint32 little-endian .bin, one shard per source file",
        "dev_set": {"repo": DEV_REPO, "revision": DEV_REVISION, "excluded": sum(m["num_excluded"] for m in used)},
        "seed": seed,
        "target_tokens": target_tokens,
        "total_tokens": total,
        "num_docs": sum(m["num_docs"] for m in used),
        "shards": used,
    }
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# Worker processes rebuild the tokenizer and dev ids once each.
_worker_settings = {}
_worker = {}


def _init_worker(settings):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(
        TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION, cache_dir=settings["cache_dir"],
    ))
    assert (tok.token_to_id("<|begin_of_text|>"), tok.token_to_id("<|end_of_text|>")) == (BOS_ID, EOS_ID)

    def tokenize(text):
        return [BOS_ID] + tok.encode(text, add_special_tokens=False).ids

    def download(filename, local_dir):
        return hf_hub_download(
            DATA_REPO, filename, repo_type="dataset", revision=DATA_REVISION, local_dir=local_dir,
        )

    _worker.update(settings=settings, tokenize=tokenize, download=download,
                   exclude_ids=load_dev_ids(DEV_REPO, revision=DEV_REVISION, filename=DEV_FILENAME,
                                            cache_dir=settings["cache_dir"]))


def _run_task(task):
    index, filename = task
    s = _worker["settings"]
    return process_file(
        index, filename, output_dir=s["output_dir"], download_dir=s["download_dir"],
        tokenize=_worker["tokenize"], exclude_ids=_worker["exclude_ids"], download=_worker["download"],
    )


def main():
    user = os.environ["USER"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=f"/work/{user}/fineweb-edu-llama3-upwash")
    parser.add_argument("--target-tokens", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--cache-dir", default=f"/work/{user}/.cache/huggingface")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _worker_settings.update(
        output_dir=str(output_dir), download_dir=str(output_dir / "raw"), cache_dir=args.cache_dir,
    )
    manifest = prepare(
        list_source_files(), output_dir=output_dir, target_tokens=int(args.target_tokens),
        seed=args.seed, run_file=None, workers=args.workers,
    )
    shutil.rmtree(output_dir / "raw", ignore_errors=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=1))


if __name__ == "__main__":
    main()
