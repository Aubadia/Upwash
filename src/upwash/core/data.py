import torch

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
            