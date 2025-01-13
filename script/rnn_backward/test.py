import torch
import torch.nn.functional as F

# y[n] = a[n] * y[n - 1] + x[n]

dtype = torch.float32

b = 2
n = 128
d = 64

b = 1
n = 2
d = 2

x = torch.randn(b, n, d, dtype=dtype).requires_grad_()
dy = torch.randn(b, n, d, dtype=dtype).requires_grad_()
log_a = (F.logsigmoid(torch.randn(b, n, d, dtype=dtype))).requires_grad_()

# bwd version1
da1 = []
ds1 = torch.zeros(b, d, dtype=dtype)

s_array = []
s = torch.zeros(b, d, dtype=dtype)
for i in range(n):
    s = torch.exp(log_a[:, i]) * s + x[:, i]
    s_array.append(s)
s = torch.stack(s_array, dim=1)
print(s.shape)

ds_array = []
for i in range(n - 1, -1, -1):
    if i == n - 1:
        ds1 = dy[:, i]
    else:
        ds1 = torch.exp(log_a[:, i + 1]) * ds1 + dy[:, i]
    ds_array.append(ds1)
    if i == 0:
        si = torch.zeros(b, d, dtype=dtype)
    else:
        si = s[:, i - 1]
    
    da1.append(ds1 * si)

da1 = torch.stack(da1[::-1], dim=1)
ds = torch.stack(ds_array, dim=1)
print(da1.shape)

# method 2 (wrong)
dlog_b_array = []
for i in range(n - 1, -1, -1):
    if i == n - 1:
        dlog_b = torch.zeros(b, d, dtype=dtype)
    else:
        # dlog_b = -x[:, i] * torch.exp(log_a[:, i + 1]) * ds[:, i + 1]
        dlog_b = x[:, i] * (dy[:, i] - ds[:, i]) + ds[:, i] * s[:, i]
    dlog_b_array.append(dlog_b)

dlog_b = torch.stack(dlog_b_array[::-1], dim=1)
print(dlog_b.shape)
dlog_a = torch.flip(
    torch.cumsum(torch.flip(dlog_b, dims=[1]), dim=1), dims=[1]
)
da2 = dlog_a * torch.exp(log_a)
print(da2.shape)

print(torch.norm(da1 - da2))

print(da1)
print(da2)