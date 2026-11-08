import torch
from huggingface_hub import snapshot_download
from pathlib import Path
import pyarrow.parquet as pq



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
            

def load_dataset(repo_id, *, revision, data_files, columns, cache_dir=None
                 , download=snapshot_download):
    local_dir = download(repo_id, repo_type="dataset", revision=revision, allow_patterns=data_files, cache_dir=cache_dir)
    files = sorted(Path(local_dir).glob(data_files))
    if not files:
        raise ValueError(f"no files match {data_files!r} in {local_dir}")

    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns):
            for row in batch.to_pylist():
                yield row
        
