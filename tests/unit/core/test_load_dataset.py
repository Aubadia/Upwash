from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from upwash.core.data import load_dataset


def write_parquet(path: Path, rows: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), path)


class FakeDownload:
    """Stands in for huggingface_hub.snapshot_download: no network, returns a local folder."""

    def __init__(self, local_dir: Path):
        self.local_dir = local_dir
        self.calls = []

    def __call__(self, repo_id, **kwargs):
        self.calls.append({"repo_id": repo_id, **kwargs})
        return str(self.local_dir)


def make_repo(root: Path) -> Path:
    # Write the later file first, so creation order differs from the expected sorted order.
    write_parquet(
        root / "sample" / "10BT" / "001.parquet",
        {"id": ["c"], "text": ["third"], "score": [0.3]},
    )
    write_parquet(
        root / "sample" / "10BT" / "000.parquet",
        {"id": ["a", "b"], "text": ["first", "second"], "score": [0.1, 0.2]},
    )
    # Outside the requested pattern: must not be read.
    write_parquet(
        root / "sample" / "100BT" / "000.parquet",
        {"id": ["z"], "text": ["other subset"], "score": [0.9]},
    )
    return root


def test_load_dataset_yields_selected_columns_from_matching_files_in_sorted_order(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))

    rows = list(
        load_dataset(
            "org/corpus",
            revision="abc123",
            data_files="sample/10BT/*.parquet",
            columns=["id", "text"],
            download=download,
        )
    )

    # Only the requested columns, only the matching files, files in sorted path order.
    assert rows == [
        {"id": "a", "text": "first"},
        {"id": "b", "text": "second"},
        {"id": "c", "text": "third"},
    ]


def test_load_dataset_reads_the_columns_the_caller_requests(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))

    rows = list(
        load_dataset(
            "org/corpus",
            revision="abc123",
            data_files="sample/10BT/*.parquet",
            columns=["text", "score"],
            download=download,
        )
    )

    # Datasets differ in their column names, so the selection must follow the caller.
    assert rows == [
        {"text": "first", "score": 0.1},
        {"text": "second", "score": 0.2},
        {"text": "third", "score": 0.3},
    ]


def test_load_dataset_raises_when_no_file_matches(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))

    # A mistyped pattern must fail loudly instead of silently yielding no rows.
    # Iterating triggers the generator's body, where the check runs.
    with pytest.raises(ValueError, match=r"sample/1BT/\*\.parquet"):
        list(
            load_dataset(
                "org/corpus",
                revision="abc123",
                data_files="sample/1BT/*.parquet",
                columns=["id", "text"],
                download=download,
            )
        )


def test_load_dataset_requests_a_pinned_dataset_snapshot_of_only_matching_files(tmp_path):
    download = FakeDownload(make_repo(tmp_path / "repo"))
    cache_dir = tmp_path / "cache"

    list(
        load_dataset(
            "org/corpus",
            revision="abc123",
            data_files="sample/10BT/*.parquet",
            columns=["id", "text"],
            cache_dir=cache_dir,
            download=download,
        )
    )

    # The revision pin and file pattern reach the downloader, so only the pinned,
    # matching files are fetched; an existing cache entry is reused by the downloader.
    assert download.calls == [
        {
            "repo_id": "org/corpus",
            "repo_type": "dataset",
            "revision": "abc123",
            "allow_patterns": "sample/10BT/*.parquet",
            "cache_dir": cache_dir,
        }
    ]
