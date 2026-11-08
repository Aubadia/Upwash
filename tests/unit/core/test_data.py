import pytest
import torch

from upwash.core.data import DataLoader, TensorDataset


def make_samples(count=5):
    # Distinct values make ordering and pairing mistakes visible.
    inputs = torch.arange(count, device="cpu", dtype=torch.float32).reshape(count, 1)
    targets = inputs * 10
    return inputs, targets


def test_tensor_dataset_returns_length_and_paired_samples():
    inputs, targets = make_samples()
    dataset = TensorDataset(inputs, targets)

    assert len(dataset) == 5
    x, y = dataset[3]
    torch.testing.assert_close(x, torch.tensor([3.0]))
    torch.testing.assert_close(y, torch.tensor([30.0]))


def test_data_loader_batches_in_order_and_keeps_last_partial_batch():
    inputs, targets = make_samples()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    batches = list(loader)

    # 5 samples with batch_size=2 -> sizes [2, 2, 1], in dataset order.
    assert len(batches) == 3
    for (batch_inputs, batch_targets), (start, stop) in zip(
        batches, [(0, 2), (2, 4), (4, 5)]
    ):
        # Samples of shape (1,) are stacked into (batch, 1).
        torch.testing.assert_close(batch_inputs, inputs[start:stop])
        torch.testing.assert_close(batch_targets, targets[start:stop])


def test_data_loader_drop_last_discards_partial_batch():
    inputs, targets = make_samples()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, drop_last=True)

    batch_sizes = [batch_inputs.shape[0] for batch_inputs, _ in loader]

    assert batch_sizes == [2, 2]


def test_data_loader_default_batch_size_is_one():
    inputs, targets = make_samples()
    loader = DataLoader(TensorDataset(inputs, targets))

    batch_shapes = [batch_inputs.shape for batch_inputs, _ in loader]

    assert batch_shapes == [torch.Size([1, 1])] * 5


@pytest.mark.parametrize(
    ("count", "drop_last", "expected_len"),
    [(5, False, 3), (5, True, 2), (4, False, 2), (4, True, 2)],
    ids=["keep-partial", "drop-partial", "divisible-keep", "divisible-drop"],
)
def test_data_loader_len_counts_batches(count, drop_last, expected_len):
    # 4 samples with batch_size=2 divide evenly: no partial batch either way.
    inputs, targets = make_samples(count)
    loader = DataLoader(
        TensorDataset(inputs, targets), batch_size=2, drop_last=drop_last
    )

    assert len(loader) == expected_len


def test_data_loader_can_be_iterated_more_than_once():
    inputs, targets = make_samples()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    first = [batch_inputs for batch_inputs, _ in loader]
    second = [batch_inputs for batch_inputs, _ in loader]

    # A generator-based loader would yield nothing the second time.
    assert len(second) == len(first) == 3
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
