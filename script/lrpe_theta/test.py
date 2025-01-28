import torch

base = 10000
head_dim = 32
h = 2

def f(d, base=1000):
    theta = base ** (
        -2 / d * torch.arange(d // 2, dtype=torch.int64)
    ).float()
    
    return theta

d = head_dim * h
theta1 = f(d, base)

d = head_dim
theta2 = f(d, base)

d = head_dim * h
theta3 = base ** (
    -2 / d * torch.arange(head_dim // 2, dtype=torch.int64)
).float()

theta4 = base ** (
    -2 / head_dim * torch.arange(d // 2, dtype=torch.int64)
).float()

print(theta1)
print(theta2)
print(theta3)
print(theta4)