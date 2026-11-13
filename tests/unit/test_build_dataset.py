import numpy as np
import pytest

from upwash.builder import build_dataset
from upwash.config import DataConfig
from upwash.core.data import DataLoader


def write_npy_folder(root, name, tokens):
    """One Flock folder: tokens.npy (int32) next to an offsets.npy that must be ignored."""
    folder = root / name
    folder.mkdir()
    np.save(folder / "tokens.npy", np.asarray(tokens, dtype=np.int32))
    np.save(folder / "offsets.npy", np.asarray([0, 3, 7], dtype=np.int64))


def write_bin(path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype="<u4").tofile(path)


def flock_config(root, num_tokens):
    return DataConfig(root=str(root), pattern="*/tokens.npy", format="npy", seq_len=4, num_tokens=num_tokens)


@pytest.fixture
def flock_root(tmp_path):
    # Written out of order: the dataset must still read folders sorted by path.
    write_npy_folder(tmp_path, "000001", range(100, 105))  # 5 tokens -> 1 sample
    write_npy_folder(tmp_path, "000000", range(0, 10))     # 10 tokens -> 2 samples
    return tmp_path


def test_build_dataset_reads_matching_npy_shards_in_sorted_order(flock_root):
    dataset = build_dataset(flock_config(flock_root, num_tokens=12))  # 12 // 4 = 3 samples

    assert len(dataset) == 3
    x0, y0 = dataset[0]
    assert x0.tolist() == [0, 1, 2, 3]
    assert y0.tolist() == [1, 2, 3, 4]
    # Sample 2 is the first sample of folder 000001; offsets.npy is not read as tokens.
    x2, y2 = dataset[2]
    assert x2.tolist() == [100, 101, 102, 103]
    assert y2.tolist() == [101, 102, 103, 104]


@pytest.mark.parametrize(
    ("num_tokens", "expected_len"),
    [(8, 2), (11, 2), (4, 1)],
    ids=["exact-two-samples", "partial-sample-dropped", "one-sample"],
)
def test_build_dataset_keeps_only_the_samples_within_the_token_budget(flock_root, num_tokens, expected_len):
    dataset = build_dataset(flock_config(flock_root, num_tokens=num_tokens))

    # num_samples = num_tokens // seq_len: never train on more than the budget.
    assert len(dataset) == expected_len
    assert dataset[-1][0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]][expected_len - 1]


def test_build_dataset_rejects_a_budget_larger_than_the_data(flock_root):
    # The shards hold 3 samples; 16 tokens would need 4.
    with pytest.raises(ValueError):
        build_dataset(flock_config(flock_root, num_tokens=16))


def test_build_dataset_rejects_a_pattern_that_matches_no_files(tmp_path):
    config = DataConfig(root=str(tmp_path), pattern="*/tokens.npy", format="npy", seq_len=4, num_tokens=4)

    # Fail loudly on a wrong root or pattern instead of training on nothing.
    with pytest.raises(ValueError):
        build_dataset(config)


def test_build_dataset_reads_bin_shards(tmp_path):
    write_bin(tmp_path / "sample" / "10BT" / "001_00000.bin", range(100, 105))
    write_bin(tmp_path / "sample" / "10BT" / "000_00000.bin", range(0, 10))
    config = DataConfig(root=str(tmp_path), pattern="sample/10BT/*.bin", format="bin", seq_len=4, num_tokens=12)

    dataset = build_dataset(config)

    assert len(dataset) == 3
    assert dataset[0][0].tolist() == [0, 1, 2, 3]
    assert dataset[2][0].tolist() == [100, 101, 102, 103]


def test_build_dataset_output_feeds_the_data_loader(flock_root):
    dataset = build_dataset(flock_config(flock_root, num_tokens=12))

    batches = list(DataLoader(dataset, batch_size=2))

    assert [tuple(x.shape) for x, _ in batches] == [(2, 4), (1, 4)]
