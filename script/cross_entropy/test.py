import torch
import torch.nn.functional as F

b = 32
n = 256
g = 1
d = 1024

x = torch.randn(b, n, g, d)
y = torch.randint(0, d, (b, n, g))

loss1 = F.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))
loss2 = F.cross_entropy(x.squeeze(-1).view(-1, x.shape[-1]), y.squeeze(-1).view(-1))

print(torch.norm(loss1 - loss2))