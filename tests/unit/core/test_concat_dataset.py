from itertools import islice

import numpy as np
import pytest

from upwash.core.data import ConcatDataset, DataLoader, PackedTokenDataset


class StrictListDataset:
    """A map-style dataset of labelled items that accepts only indices in [0, len).

    Strict on purpose: ConcatDataset must translate every global index into a
    valid local one, so a negative or out-of-range local index fails loudly.
    """

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if not 0 <= i < len(self.items):
            raise AssertionError(f"local index {i} outside [0, {len(self.items)})")
        return self.items[i]


def make_concat():
    # Lengths 2, 1, 3: boundaries at global indices 2 and 3.
    return ConcatDataset(
        [
            StrictListDataset(["a0", "a1"]),
            StrictListDataset(["b0"]),
            StrictListDataset(["c0", "c1", "c2"]),
        ]
    )


def test_concat_dataset_len_is_the_sum_of_its_parts():
    assert len(make_concat()) == 6


def test_concat_dataset_maps_global_indices_to_each_part_in_order():
    dataset = make_concat()

    assert [dataset[i] for i in range(6)] == ["a0", "a1", "b0", "c0", "c1", "c2"]


def test_concat_dataset_skips_empty_parts():
    # A shard shorter than seq_len + 1 tokens has no samples (len 0).
    dataset = ConcatDataset(
        [
            StrictListDataset(["a0", "a1"]),
            StrictListDataset([]),
            StrictListDataset(["c0"]),
        ]
    )

    assert len(dataset) == 3
    assert [dataset[i] for i in range(3)] == ["a0", "a1", "c0"]


@pytest.mark.parametrize(("index", "expected"), [(-1, "c2"), (-4, "b0"), (-6, "a0")])
def test_concat_dataset_supports_negative_indices(index, expected):
    # Like a Python sequence: dataset[-k] is dataset[len - k].
    assert make_concat()[index] == expected


@pytest.mark.parametrize("index", [6, 7, -7])
def test_concat_dataset_raises_index_error_outside_its_samples(index):
    with pytest.raises(IndexError):
        make_concat()[index]


def test_concat_dataset_iteration_stops_after_the_last_sample():
    dataset = make_concat()

    # Iteration indexes 0, 1, 2, ... until IndexError; islice turns a missing
    # IndexError into a failure instead of a hang.
    assert list(islice(iter(dataset), len(dataset) + 1)) == ["a0", "a1", "b0", "c0", "c1", "c2"]


def write_tokens(path, tokens):
    np.asarray(tokens, dtype="<u4").tofile(path)
    return path


def test_concat_dataset_reads_packed_token_shards_in_order_without_crossing_shards(tmp_path):
    shards = [
        # 10 tokens, seq_len 4 -> 2 samples; token 9 is left over and dropped.
        PackedTokenDataset(write_tokens(tmp_path / "000.bin", range(0, 10)), seq_len=4),
        # 5 tokens -> 1 sample.
        PackedTokenDataset(write_tokens(tmp_path / "001.bin", range(100, 105)), seq_len=4),
    ]

    dataset = ConcatDataset(shards)

    assert len(dataset) == 3
    x1, y1 = dataset[1]
    assert x1.tolist() == [4, 5, 6, 7]
    assert y1.tolist() == [5, 6, 7, 8]
    # The first sample of the second shard starts at that shard's first token,
    # not at the leftover token 9 of the first shard.
    x2, y2 = dataset[2]
    assert x2.tolist() == [100, 101, 102, 103]
    assert y2.tolist() == [101, 102, 103, 104]

    batches = list(DataLoader(dataset, batch_size=2))
    assert [tuple(x.shape) for x, _ in batches] == [(2, 4), (1, 4)]
