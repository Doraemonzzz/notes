import torch
from einops import repeat

shape = [1, 2, 3]
log_decay_array = torch.tensor([-0.3, -0.4, -0.5])
N = sum(pn ** 2 for pn in shape)

l = 0
array = torch.tensor([], device=torch.cuda.current_device())
mask = []

for i, n in enumerate(shape):
    log_decay = log_decay_array[i]
    m = n * n
    l += m
    array = torch.cat([log_decay + array, torch.zeros(n * n, device=torch.cuda.current_device()), ])
    pad = -float("inf") * torch.ones(N - l, device=torch.cuda.current_device())
    print(torch.cat([array, pad]).shape)
    mask.append(repeat(torch.cat([array, pad]), "l -> g l", g=m))

mask = torch.cat(mask, dim=0)

print(mask.shape)
print(mask)