import numpy as np
import pytest

from upwash.core.data import write_token_shard

# Larger than uint16's maximum (65,535), like Llama 3's <|end_of_text|> (128001),
# so a narrower dtype would corrupt it.
EOS_ID = 128001


def fake_tokenize(text):
    """Stands in for the Llama 3 tokenizer: one token per character, its code point."""
    return [ord(c) for c in text]


ROWS = [
    {"id": "a", "text": "hi"},
    {"id": "dev-1", "text": "xx"},  # in the dev set: must not reach the training data
    {"id": "b", "text": "yo!"},
]


def test_write_token_shard_writes_uint32_tokens_with_eos_after_each_kept_document(tmp_path):
    output_path = tmp_path / "train-000.bin"

    write_token_shard(
        ROWS,
        tokenize=fake_tokenize,
        eos_id=EOS_ID,
        exclude_ids={"dev-1"},
        output_path=output_path,
    )

    # "hi" -> 104, 105; "yo!" -> 121, 111, 33; each followed by EOS; "xx" excluded.
    expected = [104, 105, EOS_ID, 121, 111, 33, EOS_ID]
    assert output_path.stat().st_size == len(expected) * 4  # 4 bytes per uint32 token
    np.testing.assert_array_equal(np.fromfile(output_path, dtype="<u4"), expected)


def test_write_token_shard_returns_document_and_token_counts(tmp_path):
    metadata = write_token_shard(
        ROWS,
        tokenize=fake_tokenize,
        eos_id=EOS_ID,
        exclude_ids={"dev-1"},
        output_path=tmp_path / "train-000.bin",
    )

    # num_tokens includes the EOS tokens: it is what the 200B token budget counts against.
    assert metadata == {"num_docs": 2, "num_excluded": 1, "num_tokens": 7}


def test_write_token_shard_leaves_no_output_file_when_writing_fails(tmp_path):
    output_path = tmp_path / "train-000.bin"

    def failing_tokenize(text):
        if text == "yo!":
            raise RuntimeError("tokenizer crashed")
        return fake_tokenize(text)

    with pytest.raises(RuntimeError, match="tokenizer crashed"):
        write_token_shard(
            ROWS,
            tokenize=failing_tokenize,
            eos_id=EOS_ID,
            exclude_ids={"dev-1"},
            output_path=output_path,
        )

    # A half-written shard at the final path would look complete and be trained on.
    assert not output_path.exists()
