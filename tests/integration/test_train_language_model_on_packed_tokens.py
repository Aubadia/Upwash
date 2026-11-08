import numpy as np
import torch
import torch.nn.functional as F

from upwash.core.data import DataLoader, PackedTokenDataset
from upwash.core.trainer import Trainer

VOCAB_SIZE = 8
SEQ_LEN = 4
BATCH_SIZE = 4


def next_token_loss(logits, labels):
    """Cross-entropy over the vocabulary for every position: [B, T, V] vs [B, T]."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def mean_loss(trainer, loader):
    with torch.no_grad():
        losses = [trainer.compute_loss(x, y).item() for x, y in loader]
    return sum(losses) / len(losses)


def test_tiny_language_model_learns_next_token_from_a_packed_shard(tmp_path):
    # A repeating sequence 0..7: the next token is always (t + 1) % 8, so a model
    # that looks only at the current token can learn it exactly.
    path = tmp_path / "shard.bin"
    np.asarray(list(range(VOCAB_SIZE)) * 8 + [0], dtype=np.uint32).tofile(path)  # 65 tokens

    dataset = PackedTokenDataset(path, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # The loader stacks samples into [batch, seq_len] int64 batches.
    x, y = next(iter(loader))
    assert x.shape == (BATCH_SIZE, SEQ_LEN)
    assert y.shape == (BATCH_SIZE, SEQ_LEN)
    assert x.dtype == torch.int64

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Embedding(VOCAB_SIZE, 16),
            torch.nn.Linear(16, VOCAB_SIZE),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = Trainer(model=model, loss_fn=next_token_loss, optimizer=optimizer)

    initial_loss = mean_loss(trainer, loader)
    for _ in range(30):
        state = trainer.train(loader)
    final_loss = mean_loss(trainer, loader)

    # (65 - 1) // 4 = 16 samples -> 4 batches per pass.
    assert state.global_step == 4
    # Uniform guessing over 8 tokens costs ln 8 ~= 2.08; a learned rule costs ~0.
    assert initial_loss > 1.0
    assert final_loss < 0.05
