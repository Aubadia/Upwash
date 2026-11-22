import itertools

import torch
import lance
import numpy as np
import pyarrow as pa

def _schema(seq_len):
    return pa.schema([("tokens", pa.list_(pa.uint32(), seq_len+1))])

def _to_record_batch(chunk, seq_len, schema):
    for w in chunk:
        if len(w) != seq_len + 1:
            raise ValueError(f"window has {len(w)} tokens, expected seq_len + 1")
    
    flat = pa.array(np.stack(chunk).reshape(-1), type=pa.uint32())
    return pa.record_batch([pa.FixedSizeListArray.from_arrays(flat, seq_len+1)], schema = schema)


def write_windows(uri, windows, *, seq_len, max_rows_per_file=1024*1024, storage_options=None, batch_rows=1024):
    schema = _schema(seq_len)
    it = iter(windows)
    chunk = list(itertools.islice(it, batch_rows))
    if not chunk:
        raise ValueError("no windows to write")
    
    first = _to_record_batch(chunk, seq_len, schema)

    def batches():
        yield first
        chunk = list(itertools.islice(it, batch_rows))
        while chunk:
            yield _to_record_batch(chunk, seq_len, schema)
            chunk = list(itertools.islice(it, batch_rows))
            

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    dataset = lance.write_dataset(reader, str(uri), mode="create", max_rows_per_file=max_rows_per_file,
                        storage_options=storage_options)
    
    return dataset.count_rows()

class LanceWindows:
    def __init__(self, uri, *, version=None):
        self._dataset = lance.dataset(str(uri), version=version)
        schema = self._dataset.schema
        if "tokens" not in schema.names or not pa.types.is_fixed_size_list(schema.field("tokens").type):
            raise ValueError("expected a 'tokens' fixed_size_list<uint32>column")
        
        self.seq_len = schema.field("tokens").type.list_size - 1
        self.version = self._dataset.version
        self._rows = self._dataset.count_rows()

    def __len__(self):
        return self._rows

    def read(self, indices):
        indices = list(indices)
        for i in indices:
            if not 0 <= i < self._rows:
                raise IndexError(f"index {i} out of range of {self._rows} windows")
        table = self._dataset.take(indices, columns=["tokens"])
        chunked = table["tokens"]
        column = chunked.combine_chunks()
        flat = np.asarray(column.values, dtype=np.uint32)
        return flat.reshape(len(indices), self.seq_len + 1)
        

class LanceWindowDataset:
    def __init__(self, uri, *, version=None):
        self._windows = LanceWindows(uri, version=version)
        self.seq_len = self._windows.seq_len

    def __len__(self):
        return len(self._windows)
    
    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        
        row = self._windows.read([i])[0]
        chunk = torch.from_numpy(row.astype(np.int64))
        return chunk[:-1], chunk[1:]