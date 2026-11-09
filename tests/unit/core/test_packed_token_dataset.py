from itertools import islice

import numpy as np
import pytest
import torch

from upwash.core.data import PackedTokenDataset


def write_tokens(path, tokens):
    """Writes a token shard in the format write_token_shard produces: raw little-endian uint32 ids ("<u4")."""
    np.asarray(tokens, dtype="<u4").tofile(path)
    return path


def test_packed_token_dataset_splits_tokens_into_inputs_and_next_token_labels(tmp_path):
    path = write_tokens(tmp_path / "shard.bin", [10, 11, 12, 13, 14, 15, 16, 17, 18])

    dataset = PackedTokenDataset(path, seq_len=4)

    # Each sample reads seq_len + 1 tokens; labels are the inputs shifted by one,
    # so neighbouring samples share the boundary token (14).
    x0, y0 = dataset[0]
    x1, y1 = dataset[1]
    assert x0.tolist() == [10, 11, 12, 13]
    assert y0.tolist() == [11, 12, 13, 14]
    assert x1.tolist() == [14, 15, 16, 17]
    assert y1.tolist() == [15, 16, 17, 18]


@pytest.mark.parametrize(
    ("num_tokens", "expected_len"),
    [
        (9, 2),   # exactly 2 * seq_len + 1
        (10, 2),  # one leftover token is dropped
        (8, 1),   # 2 * seq_len is one short of a second full sample
        (5, 1),   # smallest file with one sample
        (4, 0),   # seq_len tokens leave no label for the last input
    ],
)
def test_packed_token_dataset_len_counts_only_full_samples(tmp_path, num_tokens, expected_len):
    path = write_tokens(tmp_path / "shard.bin", range(num_tokens))

    dataset = PackedTokenDataset(path, seq_len=4)

    assert len(dataset) == expected_len


def test_packed_token_dataset_returns_int64_tensors_and_keeps_large_token_ids(tmp_path):
    # 128001 is Llama 3's <|end_of_text|>, above uint16's maximum.
    path = write_tokens(tmp_path / "shard.bin", [1, 128001, 2, 128001, 3])

    x, y = PackedTokenDataset(path, seq_len=4)[0]

    # Embedding lookups and cross-entropy targets require int64 indices.
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.tolist() == [1, 128001, 2, 128001]
    assert y.tolist() == [128001, 2, 128001, 3]


@pytest.mark.parametrize("seq_len", [0, -1])
def test_packed_token_dataset_rejects_a_non_positive_seq_len(tmp_path, seq_len):
    path = write_tokens(tmp_path / "shard.bin", range(9))

    with pytest.raises(ValueError):
        PackedTokenDataset(path, seq_len=seq_len)


@pytest.mark.parametrize(("index", "same_as"), [(-1, 1), (-2, 0)])
def test_packed_token_dataset_supports_negative_indices(tmp_path, index, same_as):
    path = write_tokens(tmp_path / "shard.bin", [10, 11, 12, 13, 14, 15, 16, 17, 18])
    dataset = PackedTokenDataset(path, seq_len=4)  # len 2

    # Like a Python sequence: dataset[-k] is dataset[len - k].
    x, y = dataset[index]
    expected_x, expected_y = dataset[same_as]
    assert x.tolist() == expected_x.tolist()
    assert y.tolist() == expected_y.tolist()


@pytest.mark.parametrize("index", [2, 3, -3])
def test_packed_token_dataset_raises_index_error_outside_its_samples(tmp_path, index):
    path = write_tokens(tmp_path / "shard.bin", [10, 11, 12, 13, 14, 15, 16, 17, 18])
    dataset = PackedTokenDataset(path, seq_len=4)  # len 2

    # Without this, out-of-range reads return short or empty tensors silently.
    with pytest.raises(IndexError):
        dataset[index]


def test_packed_token_dataset_iteration_stops_after_the_last_sample(tmp_path):
    path = write_tokens(tmp_path / "shard.bin", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    dataset = PackedTokenDataset(path, seq_len=4)  # len 2; one leftover token

    # A class with only __getitem__ is iterated by indexing 0, 1, 2, ... until
    # IndexError. islice caps the read so a missing IndexError fails, not hangs.
    samples = list(islice(iter(dataset), len(dataset) + 1))

    assert len(samples) == 2
    assert samples[-1][1].tolist() == [15, 16, 17, 18]
