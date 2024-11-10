import torch
import torch.nn.functional as F

b = 32
n = 256
g = 2
d = 1024

x = torch.randn(b, n, g, d)
y = torch.randint(0, d, (b, n, g))

loss1 = F.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

loss_list = []
for i in range(g):
    loss_list.append(F.cross_entropy(x[:, :, i, :].view(-1, x.shape[-1]), y[:, :, i].view(-1)))
loss2 = torch.mean(torch.stack(loss_list, dim=0))

print(torch.norm(loss1 - loss2))