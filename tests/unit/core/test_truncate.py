from itertools import islice

import pytest

from upwash.core.data import Truncate


class StrictListDataset:
    """A map-style dataset that accepts only indices in [0, len)."""

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if not 0 <= i < len(self.items):
            raise AssertionError(f"index {i} outside [0, {len(self.items)})")
        return self.items[i]


def make_source():
    return StrictListDataset(["s0", "s1", "s2", "s3", "s4", "s5"])


def test_truncate_keeps_only_the_first_num_samples_in_order():
    dataset = Truncate(make_source(), num_samples=4)

    # A token budget of 4 * seq_len tokens: the first 4 samples, nothing more.
    assert len(dataset) == 4
    assert [dataset[i] for i in range(4)] == ["s0", "s1", "s2", "s3"]


def test_truncate_accepts_exactly_the_whole_dataset():
    dataset = Truncate(make_source(), num_samples=6)

    assert len(dataset) == 6
    assert dataset[5] == "s5"


def test_truncate_rejects_a_budget_larger_than_the_dataset():
    # Training on less data than budgeted must fail up front, not silently.
    with pytest.raises(ValueError):
        Truncate(make_source(), num_samples=7)


def test_truncate_rejects_a_negative_budget():
    with pytest.raises(ValueError):
        Truncate(make_source(), num_samples=-1)


@pytest.mark.parametrize(("index", "expected"), [(-1, "s3"), (-4, "s0")])
def test_truncate_counts_negative_indices_from_the_truncated_end(index, expected):
    dataset = Truncate(make_source(), num_samples=4)

    # dataset[-1] is the last kept sample (s3), not the source's last (s5).
    assert dataset[index] == expected


@pytest.mark.parametrize("index", [4, 5, -5])
def test_truncate_raises_index_error_beyond_the_budget(index):
    dataset = Truncate(make_source(), num_samples=4)

    # Index 4 and 5 exist in the source but are outside the budget.
    with pytest.raises(IndexError):
        dataset[index]


def test_truncate_iteration_stops_at_the_budget():
    dataset = Truncate(make_source(), num_samples=4)

    assert list(islice(iter(dataset), len(dataset) + 1)) == ["s0", "s1", "s2", "s3"]
