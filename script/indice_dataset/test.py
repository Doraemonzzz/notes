from torch.utils.data import Dataset
from distributed import enable
import torch

class IndiceDataset(Dataset):
    def __init__(self, num_sample, num_class):
        n = (num_sample + num_class - 1) // num_class
        self.indice_list = (list(range(num_class)) * n)[:num_sample]

    def __len__(self):
        return len(self.indice_list)

    def __getitem__(self, idx):
        return self.indice_list[idx]

enable(overwrite=True)

is_train = False
batch_size = 500
num_sample = 1000
num_class = 8

dataset = IndiceDataset(
    num_sample=num_sample,
    num_class=num_class,
)

sampler = torch.utils.data.DistributedSampler(
    dataset,
    num_replicas=torch.distributed.get_world_size(),
    rank=torch.distributed.get_rank(),
    shuffle=is_train,
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=batch_size,
    drop_last=is_train,
)

cnt = {}
for i in range(num_class):
    cnt[i] = 0

for data in data_loader:
    # print(data)
    for i in data:
        cnt[i.item()] += 1

print(cnt, torch.distributed.get_rank(), torch.distributed.get_world_size())


    
