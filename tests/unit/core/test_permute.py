"""Seeded sample-level shuffling (task 033, step 5).

The behavioral reference is `experiment`'s Permute: one fixed permutation of the whole
pool, the same on every rank, so RankShard and resume stay index-based.
"""
import pytest
import torch

from upwash.core.data import Permute, TensorDataset


def ids_dataset(n):
    values = torch.arange(n)
    return TensorDataset(values, values)


def order(dataset):
    return [int(dataset[i][0]) for i in range(len(dataset))]


def test_permute_is_a_seeded_permutation_of_every_sample():
    shuffled = order(Permute(ids_dataset(50), seed=42))

    assert sorted(shuffled) == list(range(50))
    assert shuffled != list(range(50))


def test_the_same_seed_gives_the_same_order_and_another_seed_does_not():
    data = ids_dataset(50)

    assert order(Permute(data, seed=42)) == order(Permute(data, seed=42))
    assert order(Permute(data, seed=42)) != order(Permute(data, seed=7))


def test_permute_keeps_the_length_and_rejects_indices_past_the_end():
    shuffled = Permute(ids_dataset(10), seed=42)

    assert len(shuffled) == 10
    with pytest.raises(IndexError):
        shuffled[10]
