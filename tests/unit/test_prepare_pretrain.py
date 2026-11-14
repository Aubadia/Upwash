"""scripts/prepare_pretrain.py with local parquet files instead of HF downloads."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_pretrain.py"
spec = importlib.util.spec_from_file_location("prepare_pretrain", SCRIPT)
prepare_pretrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_pretrain)
BOS, EOS = prepare_pretrain.BOS_ID, prepare_pretrain.EOS_ID


def make_sources(root, sizes):
    """One parquet per file; doc i of file f has text 'f i' and tokenizes to len(text) tokens."""
    sources = {}
    for f, n in enumerate(sizes):
        path = root / "src" / f"file{f}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": [f"{f}-{i}" for i in range(n)], "text": ["x" * (i + 1) for i in range(n)]}), path)
        sources[f"data/file{f}.parquet"] = path
    return sources


def runner(tmp_path, sources, exclude=frozenset()):
    downloads = []

    def download(filename, local_dir):
        local_dir.mkdir(parents=True)
        target = local_dir / "copy.parquet"
        target.write_bytes(sources[filename].read_bytes())
        downloads.append(filename)
        return target

    def run(index, filename):
        return prepare_pretrain.process_file(
            index, filename, output_dir=tmp_path / "out", download_dir=tmp_path / "out" / "raw",
            tokenize=lambda text: [BOS] + [7] * len(text), exclude_ids=exclude, download=download,
        )

    return run, downloads


def test_shard_holds_bos_text_eos_documents_without_dev_ids(tmp_path):
    sources = make_sources(tmp_path, [3])
    run, _ = runner(tmp_path, sources, exclude={"0-1"})

    meta = run(0, "data/file0.parquet")

    tokens = np.fromfile(tmp_path / "out" / "shards" / "00000.bin", dtype="<u4")
    assert tokens.tolist() == [BOS, 7, EOS, BOS, 7, 7, 7, EOS]  # docs 0 and 2; doc 1 excluded
    assert meta == {"index": 0, "source": "data/file0.parquet", "num_docs": 2, "num_excluded": 1, "num_tokens": 8}
    assert not (tmp_path / "out" / "raw" / "00000").exists()  # raw parquet removed


def test_finished_shards_are_reused(tmp_path):
    sources = make_sources(tmp_path, [2])
    run, downloads = runner(tmp_path, sources)
    first = run(0, "data/file0.parquet")
    second = run(0, "data/file0.parquet")
    assert first == second and downloads == ["data/file0.parquet"]


def test_pool_is_a_seeded_prefix_that_reaches_the_target(tmp_path):
    sources = make_sources(tmp_path, [4, 4, 4, 4, 4])  # 4 docs -> 2+3+4+5 + 4 EOS = 18 tokens per file
    run, _ = runner(tmp_path, sources)
    order = prepare_pretrain.shuffled_files(list(sources), seed=3)
    # Simulate a worker that ran ahead and finished a shard past the prefix.
    run(4, order[4])

    manifest = prepare_pretrain.prepare(
        list(sources), output_dir=tmp_path / "out", target_tokens=40, seed=3, run_file=run,
    )

    assert [m["source"] for m in manifest["shards"]] == order[:3]
    assert manifest["total_tokens"] == 54 and manifest["seed"] == 3
    assert sorted(p.name for p in (tmp_path / "out" / "shards").iterdir()) == [
        "00000.bin", "00000.json", "00001.bin", "00001.json", "00002.bin", "00002.json",
    ]
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["total_tokens"] == 54


def test_file_order_depends_only_on_the_seed():
    files = [f"data/f{i}.parquet" for i in range(20)]
    assert prepare_pretrain.shuffled_files(files, 0) == prepare_pretrain.shuffled_files(list(reversed(files)), 0)
    assert prepare_pretrain.shuffled_files(files, 0) != prepare_pretrain.shuffled_files(files, 1)
    assert sorted(prepare_pretrain.shuffled_files(files, 0)) == sorted(files)
