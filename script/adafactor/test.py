import torch

n = 10
m = 10

x = torch.tensor(
    [[100, 118, 30],
     [19,  1,  30],
     [1,   1,   30]]
) ** 0.5

x2 = x ** 2

print("order 1")
print(x)
print("order 2")
print(x **2)

L = torch.einsum("d e, f e -> d f", x, x)
R = torch.einsum("d e, d f -> e f", x, x)
print("shampoo")
y0 = (L ** -0.25) * x * (R ** -0.25)
print(y0)

L = torch.sum(x2, dim=-1, keepdim=True)
R = torch.sum(x2, dim=-2, keepdim=True)

print("shampoo diag")
y1 = (L ** -0.25) * x * (R ** -0.25)
print(y1)

print("shampoo diag L")
y2 = (L ** -0.5) * x
print(y2)

print("shampoo diag R")
y3 = x * (R ** -0.5)
print(y3)

print("adafactor")
S = torch.sum(x2)
y4 = (L ** -0.5) * x * (R ** -0.5) * (S ** 0.5)
print(y4)

