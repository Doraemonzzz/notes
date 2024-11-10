import torch
import torch.nn.functional as F

n = 128
d = 64

index = torch.arange(1, n + 1)

alpha = F.logsigmoid(torch.rand(d, n))
alpha_cumsum = alpha.cumsum(dim=-1)

beta = alpha_cumsum / index

gamma = torch.cat([alpha[..., 0:1], (alpha[..., 1:] - beta[..., :-1]) / index[1:]], dim=-1)
gamma_cumsum = gamma.cumsum(dim=-1)

print(torch.norm(beta - gamma_cumsum))