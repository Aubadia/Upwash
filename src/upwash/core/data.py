import torch
from huggingface_hub import snapshot_download
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import json



class TensorDataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        self.length = len(self.inputs)
        return self.length

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]

class DataLoader:
    def __init__(self, dataset, batch_size = 1, drop_last = False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset


    def __iter__(self):
        for i in range(len(self)):
            start = i * self.batch_size
            end = min((i+1)*self.batch_size, len(self.dataset))

            xs = []
            ys = []
            for index in range(start, end):
                x, y = self.dataset[index]
                xs.append(x)
                ys.append(y)
            
            batch_inputs = torch.stack(xs)
            batch_targers = torch.stack(ys)
            yield batch_inputs, batch_targers


    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size -1) // self.batch_size


def read_parquet_rows(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            yield row

class PackedTokenDataset:
    def __init__(self, path, seq_len):
        self.seq_len = seq_len
        self.tokens=np.memmap(path, dtype="<u4", mode="r")

    def __len__(self):
        return (len(self.tokens) -1 ) //self.seq_len
    
    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = self.tokens[start: start+ self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]


def load_dataset(repo_id, *, revision, data_files, columns, cache_dir=None
                 , download=snapshot_download):
    local_dir = download(repo_id, repo_type="dataset", revision=revision, allow_patterns=data_files, cache_dir=cache_dir)
    files = sorted(Path(local_dir).glob(data_files))
    if not files:
        raise ValueError(f"no files match {data_files!r} in {local_dir}")

    for path in files:
        yield from read_parquet_rows(path, columns)

    
def load_dev_ids(repo_id, *, revision, filename, cache_dir=None, download=snapshot_download) -> set[str]:
    local_dir = download(repo_id, repo_type="dataset", revision=revision, allow_patterns=filename, cache_dir=cache_dir)
    text = (Path(local_dir) / filename).read_text()
    lines = text.splitlines()
    ids = set()
    for line in lines:
        if line:
            ids.add(line)
    return ids


def prepare_token_shards(repo_id, *, revision, data_files, output_dir, tokenize, eos_id, 
                         exclude_ids, cache_dir=None, download=snapshot_download) -> dict[str, int]:
    local_dir = download(repo_id, repo_type="dataset", revision=revision, allow_patterns=data_files, cache_dir=cache_dir)
    files = sorted(Path(local_dir).glob(data_files))

    total = {"num_files": 0, "num_docs": 0, "num_excluded": 0, "num_tokens": 0}

    for path in files:
        rel = path.relative_to(local_dir)
        bin_path = Path(output_dir) / rel.with_suffix(".bin")
        json_path = bin_path.with_suffix(".json")

        if not json_path.exists():
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            rows = read_parquet_rows(path, ["id", "text"])
            meta = write_token_shard(rows, tokenize=tokenize, eos_id=eos_id, exclude_ids=exclude_ids, output_path=bin_path)
            json_path.write_text(json.dumps(meta))

        meta = json.loads(json_path.read_text())
        total["num_files"] += 1
        total["num_excluded"] += meta["num_excluded"]
        total["num_docs"] += meta["num_docs"]
        total["num_tokens"] += meta["num_tokens"]

    return total


def write_token_shard(rows, *, tokenize, eos_id, exclude_ids, output_path) -> dict[str, int]:
    tmp_path = Path(str(output_path) + ".tmp")
    num_docs = 0
    num_excluded = 0
    num_tokens = 0

    with open(tmp_path, "wb") as f:
        for row in rows:
            if row["id"] in exclude_ids:
                num_excluded += 1
                continue
            ids = tokenize(row["text"]) +[eos_id]
            data = np.asarray(ids, dtype="<u4").tobytes()
            f.write(data)
            num_docs+= 1
            num_tokens += len(ids)
    
    tmp_path.replace(output_path)
    
    return {"num_docs": num_docs, "num_excluded": num_excluded, "num_tokens": num_tokens}