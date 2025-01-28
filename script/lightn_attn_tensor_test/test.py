import torch

b, h, n, d, e = 2, 12, 1, 64, 32
device = 'cuda'


alpha = torch.randn(b, h, d).to(device)
beta = torch.randn(b, h, e).to(device)
s = torch.randn(b, h, d, e).to(device)
do = torch.randn(b, h, e).to(device)
q = torch.randn(b, h, d).to(device)

# method1
t1 = torch.einsum('b h d, b h e, b h d e -> b h d e', alpha, beta, s)
t2 = torch.einsum('b h d e, b h e -> b h d', t1, do)
o1 = q * t2

# method2
t3 = torch.einsum('b h d, b h e, b h d e -> b h d e', q, do, s)
t4 = torch.einsum('b h d e, b h e -> b h d', t3, beta)
o2 = alpha * t4

print(torch.norm(o1 - o2))
