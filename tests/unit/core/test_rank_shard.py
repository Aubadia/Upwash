"""Per-rank sharding for DDP (task 026).

Global batch g holds samples [g*W*B, (g+1)*W*B); rank r takes the r-th block of B.
Joining the ranks' batches of one step in rank order therefore gives exactly the
batch a single process would read with batch size W*B, which is what makes W ranks
of batch B equivalent to one process of batch W*B.
"""
import pytest
import torch

from upwash.core.data import DataLoader, RankShard, TensorDataset


def ids_dataset(n):
    values = torch.arange(n)
    return TensorDataset(values, values)


def ids(dataset):
    return [int(dataset[i][0]) for i in range(len(dataset))]


def test_each_rank_takes_its_block_of_every_global_batch():
    data = ids_dataset(12)  # 2 ranks x batch 3 -> two global batches of 6

    shards = [ids(RankShard(data, rank=r, world_size=2, batch_size=3)) for r in range(2)]

    assert shards == [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]


def test_ranks_have_equal_length_and_cover_every_sample_once():
    data = ids_dataset(24)

    shards = [RankShard(data, rank=r, world_size=4, batch_size=2) for r in range(4)]

    # DDP needs the same number of updates on every rank, or the others hang.
    assert [len(s) for s in shards] == [6, 6, 6, 6]
    assert sorted(i for s in shards for i in ids(s)) == list(range(24))


def test_joined_rank_batches_equal_the_single_process_global_batch():
    data = ids_dataset(24)
    world_size, batch_size = 4, 2

    per_rank = [list(DataLoader(RankShard(data, rank=r, world_size=world_size,
                                          batch_size=batch_size), batch_size=batch_size))
                for r in range(world_size)]
    single = list(DataLoader(data, batch_size=world_size * batch_size))

    assert [len(batches) for batches in per_rank] == [len(single)] * world_size
    for step, (inputs, targets) in enumerate(single):
        torch.testing.assert_close(torch.cat([per_rank[r][step][0] for r in range(world_size)]), inputs)
        torch.testing.assert_close(torch.cat([per_rank[r][step][1] for r in range(world_size)]), targets)


def test_one_rank_is_the_dataset_itself():
    assert ids(RankShard(ids_dataset(6), rank=0, world_size=1, batch_size=3)) == list(range(6))


def test_a_partial_global_batch_is_rejected():
    # 13 samples do not fill whole global batches of 6: one rank would get an extra batch.
    with pytest.raises(ValueError, match="global batch"):
        RankShard(ids_dataset(13), rank=0, world_size=2, batch_size=3)


@pytest.mark.parametrize("world_size", [0, -1])
def test_a_non_positive_world_size_is_rejected_by_name(world_size):
    # Checked before the rank, so the error names the real problem rather than
    # reporting "rank must be in [0, 0)".
    with pytest.raises(ValueError, match="world"):
        RankShard(ids_dataset(12), rank=0, world_size=world_size, batch_size=3)


@pytest.mark.parametrize("batch_size", [0, -3])
def test_a_non_positive_batch_size_is_rejected_before_any_division(batch_size):
    with pytest.raises(ValueError, match="batch"):
        RankShard(ids_dataset(12), rank=0, world_size=2, batch_size=batch_size)


@pytest.mark.parametrize("rank", [-1, 2])
def test_a_rank_outside_the_world_is_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        RankShard(ids_dataset(12), rank=rank, world_size=2, batch_size=3)


def test_indexing_supports_negative_indices_and_rejects_out_of_range():
    shard = RankShard(ids_dataset(12), rank=1, world_size=2, batch_size=3)

    assert int(shard[-1][0]) == 11
    assert int(shard[-6][0]) == 3
    with pytest.raises(IndexError):
        shard[6]
    with pytest.raises(IndexError):
        shard[-7]
