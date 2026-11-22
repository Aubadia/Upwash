"""Training samples read from a Lance token-window dataset (task 033, step 5).

A row holds seq_len + 1 tokens; sample i is (row[:-1], row[1:]) as int64 tensors, the same
contract as PackedTokenDataset. `build_dataset` reaches it through DataConfig(format="lance"),
where `root` is the dataset URI, optionally pinned to `version`, shuffled by `shuffle_seed`
before the token budget is applied.
"""
import numpy as np
import pytest
import torch

from upwash.builder import build_dataset
from upwash.config import DataConfig
from upwash.core.data import DataLoader, RankShard
from upwash.core.lance_windows import LanceWindowDataset, write_windows

S = 4


def write(uri, n, seq_len=S):
    windows = [np.arange(i * 100, i * 100 + seq_len + 1, dtype=np.uint32) for i in range(n)]
    write_windows(uri, windows, seq_len=seq_len)
    return windows


def lance_config(uri, num_tokens, **overrides):
    return DataConfig(root=str(uri), pattern="", format="lance", seq_len=S, num_tokens=num_tokens, **overrides)


def test_a_sample_is_the_window_shifted_by_one_token(tmp_path):
    windows = write(tmp_path / "w.lance", 3)

    dataset = LanceWindowDataset(tmp_path / "w.lance")
    inputs, targets = dataset[1]

    assert len(dataset) == 3
    assert inputs.dtype == targets.dtype == torch.int64
    assert inputs.tolist() == windows[1][:-1].tolist()
    assert targets.tolist() == windows[1][1:].tolist()


def test_negative_and_out_of_range_indices(tmp_path):
    windows = write(tmp_path / "w.lance", 3)
    dataset = LanceWindowDataset(tmp_path / "w.lance")

    assert dataset[-1][0].tolist() == windows[2][:-1].tolist()
    with pytest.raises(IndexError):
        dataset[3]


def test_the_data_loader_batches_lance_samples(tmp_path):
    write(tmp_path / "w.lance", 5)

    batches = list(DataLoader(LanceWindowDataset(tmp_path / "w.lance"), batch_size=2))

    assert [tuple(x.shape) for x, _ in batches] == [(2, S), (2, S), (1, S)]


def test_build_dataset_reads_lance_within_the_token_budget(tmp_path):
    windows = write(tmp_path / "w.lance", 10)

    dataset = build_dataset(lance_config(tmp_path / "w.lance", num_tokens=3 * S))

    assert len(dataset) == 3
    assert [dataset[i][0].tolist() for i in range(3)] == [w[:-1].tolist() for w in windows[:3]]


def test_build_dataset_rejects_a_budget_larger_than_the_lance_data(tmp_path):
    write(tmp_path / "w.lance", 3)

    with pytest.raises(ValueError):
        build_dataset(lance_config(tmp_path / "w.lance", num_tokens=4 * S))


def test_build_dataset_rejects_a_seq_len_that_differs_from_the_dataset(tmp_path):
    write(tmp_path / "w.lance", 3, seq_len=8)

    with pytest.raises(ValueError, match="seq_len"):
        build_dataset(lance_config(tmp_path / "w.lance", num_tokens=S))


def test_build_dataset_pins_the_lance_version(tmp_path):
    import lance
    import pyarrow as pa

    windows = write(tmp_path / "w.lance", 3)
    extra = np.full((2, S + 1), 7, dtype=np.uint32)
    lance.write_dataset(pa.table({"tokens": pa.FixedSizeListArray.from_arrays(
        pa.array(extra.reshape(-1), type=pa.uint32()), S + 1)}), str(tmp_path / "w.lance"), mode="append")

    pinned = build_dataset(lance_config(tmp_path / "w.lance", num_tokens=3 * S, version=1))
    latest = LanceWindowDataset(tmp_path / "w.lance")

    assert len(latest) == 5
    assert [pinned[i][0].tolist() for i in range(3)] == [w[:-1].tolist() for w in windows]


def test_a_shuffled_lance_budget_is_a_prefix_of_the_same_permutation(tmp_path):
    write(tmp_path / "w.lance", 20)

    def first_tokens(num_samples):
        data = build_dataset(lance_config(tmp_path / "w.lance", num_tokens=num_samples * S, shuffle_seed=42))
        return [int(data[i][0][0]) for i in range(len(data))]

    short, long = first_tokens(5), first_tokens(15)

    assert long[:5] == short
    assert long != sorted(long)
    assert first_tokens(15) == long  # deterministic


def test_rank_shards_partition_a_lance_dataset(tmp_path):
    write(tmp_path / "w.lance", 8)
    data = build_dataset(lance_config(tmp_path / "w.lance", num_tokens=8 * S))

    shards = [[int(RankShard(data, rank=r, world_size=2, batch_size=2)[i][0][0]) for i in range(4)]
              for r in range(2)]

    assert shards == [[0, 100, 400, 500], [200, 300, 600, 700]]
