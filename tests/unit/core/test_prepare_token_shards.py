import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from upwash.core.data import load_dev_ids, prepare_token_shards

EOS_ID = 128001


def fake_tokenize(text):
    """Stands in for the Llama 3 tokenizer: one token per character, its code point."""
    return [ord(c) for c in text]


class FakeDownload:
    """Stands in for huggingface_hub.snapshot_download: no network, returns a local folder."""

    def __init__(self, local_dir: Path):
        self.local_dir = local_dir
        self.calls = []

    def __call__(self, repo_id, **kwargs):
        self.calls.append({"repo_id": repo_id, **kwargs})
        return str(self.local_dir)


class RecordingTokenize:
    """Tokenizes like fake_tokenize and records every text it was asked to tokenize."""

    def __init__(self):
        self.texts = []

    def __call__(self, text):
        self.texts.append(text)
        return fake_tokenize(text)


def write_parquet(path: Path, rows: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), path)


# ---------------------------------------------------------------------------
# load_dev_ids
# ---------------------------------------------------------------------------


def test_load_dev_ids_reads_one_id_per_line_from_a_pinned_dataset_file(tmp_path):
    repo = tmp_path / "dev-repo"
    repo.mkdir()
    (repo / "dev-ids.txt").write_text("dev-1\ndev-2\n\ndev-3\n")
    download = FakeDownload(repo)
    cache_dir = tmp_path / "cache"

    dev_ids = load_dev_ids(
        "org/dev-set",
        revision="def456",
        filename="dev-ids.txt",
        cache_dir=cache_dir,
        download=download,
    )

    # A set, so `row["id"] in dev_ids` is fast for 5,000 ids; blank lines are not ids.
    assert dev_ids == {"dev-1", "dev-2", "dev-3"}
    # Only the pinned file is requested, so the excluded set cannot drift between runs.
    assert download.calls == [
        {
            "repo_id": "org/dev-set",
            "repo_type": "dataset",
            "revision": "def456",
            "allow_patterns": "dev-ids.txt",
            "cache_dir": cache_dir,
        }
    ]


# ---------------------------------------------------------------------------
# prepare_token_shards
# ---------------------------------------------------------------------------


def make_repo(root: Path) -> Path:
    write_parquet(
        root / "sample" / "10BT" / "000.parquet",
        {"id": ["a", "dev-1"], "text": ["hi", "xx"]},  # dev-1 is in the dev set
    )
    write_parquet(
        root / "sample" / "10BT" / "001.parquet",
        {"id": ["b"], "text": ["yo!"]},
    )
    # Outside the requested pattern: must not become a shard.
    write_parquet(
        root / "sample" / "100BT" / "000.parquet",
        {"id": ["z"], "text": ["other subset"]},
    )
    return root


def prepare(tmp_path, download, tokenize):
    return prepare_token_shards(
        "org/corpus",
        revision="abc123",
        data_files="sample/10BT/*.parquet",
        output_dir=tmp_path / "tokens",
        tokenize=tokenize,
        eos_id=EOS_ID,
        exclude_ids={"dev-1"},
        download=download,
    )


def test_prepare_token_shards_writes_one_shard_per_matching_parquet_file(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))
    output_dir = tmp_path / "tokens"  # does not exist yet

    prepare(tmp_path, download, fake_tokenize)

    # Each shard mirrors its parquet file's path under output_dir, with a .bin suffix.
    np.testing.assert_array_equal(
        np.fromfile(output_dir / "sample" / "10BT" / "000.bin", dtype=np.uint32),
        [104, 105, EOS_ID],  # "hi"; "xx" excluded
    )
    np.testing.assert_array_equal(
        np.fromfile(output_dir / "sample" / "10BT" / "001.bin", dtype=np.uint32),
        [121, 111, 33, EOS_ID],  # "yo!"
    )
    assert not (output_dir / "sample" / "100BT").exists()
    # Every download request uses the pinned revision.
    assert download.calls
    assert all(call["revision"] == "abc123" for call in download.calls)


def test_prepare_token_shards_records_counts_per_shard_and_in_total(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))
    output_dir = tmp_path / "tokens"

    summary = prepare(tmp_path, download, fake_tokenize)

    # A sidecar .json next to each .bin holds that shard's write_token_shard counts.
    shard_dir = output_dir / "sample" / "10BT"
    assert json.loads((shard_dir / "000.json").read_text()) == {
        "num_docs": 1,
        "num_excluded": 1,
        "num_tokens": 3,
    }
    assert json.loads((shard_dir / "001.json").read_text()) == {
        "num_docs": 1,
        "num_excluded": 0,
        "num_tokens": 4,
    }
    assert summary == {"num_files": 2, "num_docs": 2, "num_excluded": 1, "num_tokens": 7}


def test_prepare_token_shards_skips_completed_shards_on_a_second_run(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))
    first = prepare(tmp_path, download, fake_tokenize)

    tokenize = RecordingTokenize()
    second = prepare(tmp_path, download, tokenize)

    # Re-running must not redo hours of tokenization, yet still report the full totals.
    assert tokenize.texts == []
    assert second == first


def test_prepare_token_shards_resumes_after_a_crash_without_redoing_finished_shards(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))
    output_dir = tmp_path / "tokens"

    def crashing_tokenize(text):
        if text == "yo!":  # first document of the second file
            raise RuntimeError("tokenizer crashed")
        return fake_tokenize(text)

    with pytest.raises(RuntimeError, match="tokenizer crashed"):
        prepare(tmp_path, download, crashing_tokenize)

    tokenize = RecordingTokenize()
    summary = prepare(tmp_path, download, tokenize)

    # 000 finished before the crash and is kept; only 001 is tokenized again.
    assert tokenize.texts == ["yo!"]
    np.testing.assert_array_equal(
        np.fromfile(output_dir / "sample" / "10BT" / "001.bin", dtype=np.uint32),
        [121, 111, 33, EOS_ID],
    )
    assert summary == {"num_files": 2, "num_docs": 2, "num_excluded": 1, "num_tokens": 7}
