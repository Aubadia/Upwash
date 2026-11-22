"""Tokenized documents to fixed-length training windows (task 033, step 2).

Agreed 2026-11-21:
- A document is encoded as [BOS] + tokenize(text) + [EOS]. `tokenize` is a
  str -> list[int] callable (the write_token_shard convention) that adds no special
  tokens itself.
- Documents are joined into one stream per call. Window k is stream[k*S : k*S + S + 1],
  the PackedTokenDataset slicing, so consecutive windows share one token and every
  token after the first is a target exactly once.
- Each window is one uint32 array of S + 1 tokens (one Lance row, option A). The final
  partial window is dropped; callers derive the count as N - (W*S + 1).
"""
import itertools

import numpy as np
import pytest

from upwash.prep.pack import encode_document, pack_windows

BOS, EOS = 128000, 128001


def fake_tokenize(text):
    return [ord(ch) for ch in text]


def windows(documents, seq_len):
    return [w.tolist() for w in pack_windows(documents, seq_len=seq_len)]


def test_a_document_is_bos_text_tokens_eos():
    assert encode_document("ab", fake_tokenize, bos_id=BOS, eos_id=EOS) == [BOS, 97, 98, EOS]


def test_an_empty_document_still_has_its_boundaries():
    assert encode_document("", fake_tokenize, bos_id=BOS, eos_id=EOS) == [BOS, EOS]


def test_encode_calls_tokenize_once_on_the_exact_text():
    seen = []

    def tokenize(text):
        seen.append(text)
        return [7, 8]

    assert encode_document("  keep spaces\n", tokenize, bos_id=1, eos_id=2) == [1, 7, 8, 2]
    assert seen == ["  keep spaces\n"]


def test_consecutive_windows_share_one_token():
    # S = 4 over tokens 0..12: [0..4], [4..8], [8..12].
    assert windows([list(range(13))], seq_len=4) == [
        [0, 1, 2, 3, 4], [4, 5, 6, 7, 8], [8, 9, 10, 11, 12]]


def test_windows_run_across_document_boundaries():
    assert windows([[0, 1], [2, 3, 4], [5, 6, 7, 8]], seq_len=4) == [
        [0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]


def test_a_window_needs_seq_len_plus_one_tokens():
    assert windows([[0, 1, 2, 3]], seq_len=4) == []
    assert windows([[0, 1, 2, 3, 4]], seq_len=4) == [[0, 1, 2, 3, 4]]
    assert windows([], seq_len=4) == []
    assert windows([[], []], seq_len=4) == []


def test_the_final_partial_window_is_dropped():
    # 11 tokens: windows [0..4], [4..8]; tokens 9 and 10 are dropped.
    assert windows([list(range(11))], seq_len=4) == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]


def test_windows_are_uint32_arrays_and_keep_large_ids():
    out = list(pack_windows([[BOS, 5, 6, 7, EOS, BOS]], seq_len=4))

    assert len(out) == 1
    assert isinstance(out[0], np.ndarray)
    assert out[0].dtype == np.uint32
    assert out[0].shape == (5,)
    assert out[0].tolist() == [BOS, 5, 6, 7, EOS]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_windows_reassemble_the_stream_and_account_for_every_token(seed):
    rng = np.random.default_rng(seed)
    seq_len = 8
    docs = [rng.integers(0, 1000, size=n).tolist() for n in rng.integers(0, 30, size=40)]
    stream = [t for d in docs for t in d]

    out = list(pack_windows(docs, seq_len=seq_len))

    n, w = len(stream), len(out)
    assert w == max(0, (n - 1) // seq_len)
    dropped = n - (w * seq_len + 1) if w else n
    assert 0 <= dropped < seq_len + 1
    if w:
        joined = out[0].tolist() + [t for x in out[1:] for t in x.tolist()[1:]]
        assert joined == stream[: w * seq_len + 1]


def test_packing_is_lazy_over_a_one_pass_stream():
    # Training data is streamed: the first window must come before all input is read.
    docs = ([i] * 3 for i in itertools.count())

    first = next(iter(pack_windows(docs, seq_len=4)))

    assert first.tolist() == [0, 0, 0, 1, 1]


@pytest.mark.parametrize("seq_len", [0, -4])
def test_seq_len_must_be_positive(seq_len):
    with pytest.raises(ValueError, match="seq_len"):
        list(pack_windows([[0, 1, 2]], seq_len=seq_len))
