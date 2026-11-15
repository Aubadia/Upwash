"""Sample-level shuffling and per-rank sharding of packed datasets."""

import numpy as np
import pytest
import torch

from upwash.builder import build_dataset
from upwash.config import DataConfig
from upwash.core.data import Permute, RankShard, TensorDataset


def ids_dataset(n):
    values = torch.arange(n)
    return TensorDataset(values, values)


def first_inputs(dataset):
    return [int(dataset[i][0]) for i in range(len(dataset))]


def test_permute_is_a_seeded_permutation():
    data = ids_dataset(50)

    order = first_inputs(Permute(data, seed=7))

    assert sorted(order) == list(range(50))
    assert order != list(range(50))
    assert order == first_inputs(Permute(data, seed=7))
    assert order != first_inputs(Permute(data, seed=8))


def test_shuffled_runs_read_a_prefix_of_the_same_pool(tmp_path):
    # 40 samples of 4 tokens; each sample's first token identifies it.
    np.save(tmp_path / "tokens.npy", np.arange(161, dtype=np.int32))

    def first_tokens(num_tokens, seed=0):
        data = build_dataset(DataConfig(str(tmp_path), "tokens.npy", "npy", 4, num_tokens, shuffle_seed=seed))
        return [int(data[i][0][0]) for i in range(len(data))]

    short, long = first_tokens(4 * 10), first_tokens(4 * 30)

    assert long[:10] == short
    assert len(set(long)) == 30
    assert long != sorted(long)
    unshuffled = build_dataset(DataConfig(str(tmp_path), "tokens.npy", "npy", 4, 4 * 10))
    assert [int(unshuffled[i][0][0]) for i in range(10)] == [4 * i for i in range(10)]


def test_rank_shards_partition_each_global_batch_in_rank_order():
    data = ids_dataset(12)  # 2 ranks x batch 3 -> two global batches

    shards = [first_inputs(RankShard(data, rank=r, world_size=2, batch_size=3)) for r in range(2)]

    assert shards == [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]


def test_rank_shard_rejects_a_partial_global_batch():
    with pytest.raises(ValueError, match="global batches of 6"):
        RankShard(ids_dataset(13), rank=0, world_size=2, batch_size=3)


def test_rank_shard_rejects_an_invalid_rank():
    with pytest.raises(ValueError, match="rank"):
        RankShard(ids_dataset(12), rank=2, world_size=2, batch_size=3)


def test_rank_shard_index_bounds():
    shard = RankShard(ids_dataset(12), rank=1, world_size=2, batch_size=3)
    assert int(shard[-1][0]) == 11
    with pytest.raises(IndexError):
        shard[6]
