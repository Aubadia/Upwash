"""Packed token windows stored as a Lance dataset (task 028, step 3).

One row is one training window: a `tokens` column of type
fixed_size_list<uint32, seq_len + 1> (option A, agreed 2026-11-21). The same module
writes the format (preprocessing jobs) and reads it (training), so both sides agree on
the schema.

These tests use a local directory. Lance reads S3 URIs through the same API, and
Flock S3 was checked separately (Slurm 820790: write, round trip and delete).

Behavior measured on Lance 14 before writing these tests: `take` keeps the requested order
and repeats, and an out-of-range or negative index raises OSError or OverflowError with an
unclear message. `read` therefore checks bounds itself. Creating an existing dataset
raises OSError.
"""
import numpy as np
import pytest

from upwash.core.lance_windows import LanceWindows, write_windows
from upwash.prep.pack import pack_windows

S = 4


def make_windows(n, seq_len=S):
    return [np.arange(i * 100, i * 100 + seq_len + 1, dtype=np.uint32) for i in range(n)]


def test_windows_round_trip_in_order(tmp_path):
    windows = make_windows(10)

    assert write_windows(tmp_path / "w.lance", iter(windows), seq_len=S) == 10
    table = LanceWindows(tmp_path / "w.lance")

    assert len(table) == 10
    assert table.seq_len == S
    out = table.read(list(range(10)))
    assert out.dtype == np.uint32 and out.shape == (10, S + 1)
    np.testing.assert_array_equal(out, np.stack(windows))


def test_read_keeps_the_requested_order_and_repeats(tmp_path):
    windows = make_windows(10)
    write_windows(tmp_path / "w.lance", windows, seq_len=S)

    out = LanceWindows(tmp_path / "w.lance").read([7, 2, 7, 0])

    np.testing.assert_array_equal(out, np.stack([windows[7], windows[2], windows[7], windows[0]]))


@pytest.mark.parametrize("index", [10, -1])
def test_read_rejects_indices_outside_the_table(tmp_path, index):
    write_windows(tmp_path / "w.lance", make_windows(10), seq_len=S)

    with pytest.raises(IndexError, match="10"):
        LanceWindows(tmp_path / "w.lance").read([0, index])


def test_reading_no_indices_gives_an_empty_batch(tmp_path):
    write_windows(tmp_path / "w.lance", make_windows(3), seq_len=S)

    out = LanceWindows(tmp_path / "w.lance").read([])

    assert out.dtype == np.uint32 and out.shape == (0, S + 1)


def test_many_files_still_read_as_one_table(tmp_path):
    windows = make_windows(25)
    write_windows(tmp_path / "w.lance", windows, seq_len=S, max_rows_per_file=4)

    table = LanceWindows(tmp_path / "w.lance")

    assert len(table) == 25
    np.testing.assert_array_equal(table.read([24, 0, 13]), np.stack([windows[24], windows[0], windows[13]]))


def test_writing_streams_from_pack_windows_and_keeps_large_ids(tmp_path):
    documents = ([128000] + [i] * 6 + [128001] for i in range(50))  # a one-pass generator
    expected = [w.copy() for w in pack_windows(([128000] + [i] * 6 + [128001] for i in range(50)), seq_len=S)]

    rows = write_windows(tmp_path / "w.lance", pack_windows(documents, seq_len=S), seq_len=S)

    table = LanceWindows(tmp_path / "w.lance")
    assert rows == len(expected) == len(table)
    np.testing.assert_array_equal(table.read(list(range(rows))), np.stack(expected))


def test_a_window_of_the_wrong_length_is_rejected(tmp_path):
    windows = make_windows(3) + [np.zeros(S, dtype=np.uint32)]

    with pytest.raises(ValueError, match="seq_len"):
        write_windows(tmp_path / "w.lance", windows, seq_len=S)


def test_writing_no_windows_is_an_error(tmp_path):
    # A job that produced nothing is a pipeline bug; refuse instead of creating an empty table.
    with pytest.raises(ValueError, match="no windows"):
        write_windows(tmp_path / "w.lance", [], seq_len=S)


def test_an_existing_dataset_is_never_overwritten(tmp_path):
    write_windows(tmp_path / "w.lance", make_windows(3), seq_len=S)

    with pytest.raises(OSError):
        write_windows(tmp_path / "w.lance", make_windows(5), seq_len=S)
    assert len(LanceWindows(tmp_path / "w.lance")) == 3


def test_opening_a_table_without_the_window_column_is_rejected(tmp_path):
    import lance
    import pyarrow as pa

    lance.write_dataset(pa.table({"text": ["a", "b"]}), tmp_path / "text.lance")

    with pytest.raises(ValueError, match="tokens"):
        LanceWindows(tmp_path / "text.lance")


def test_a_pinned_version_is_read_and_reported(tmp_path):
    write_windows(tmp_path / "w.lance", make_windows(3), seq_len=S)

    table = LanceWindows(tmp_path / "w.lance", version=1)

    assert table.version == 1
    assert len(table) == 3
