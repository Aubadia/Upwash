import numpy as np


def encode_document(text, tokenize, *, bos_id, eos_id):
    return [bos_id] + list(tokenize(text)) + [eos_id]

def pack_windows(documents, *, seq_len):
    if seq_len <= 0:
        raise ValueError(f"seq_len should be greater than 0, got {seq_len}")
    
    buffer = []
    for doc in documents:
        buffer.extend(doc)

        while len(buffer) >= seq_len + 1:
            yield np.asarray(buffer[: seq_len + 1], dtype=np.uint32)
            del buffer[: seq_len]