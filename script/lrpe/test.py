import torch

b = 2
b = 1
# h = 12
h = 1
# n = 128
n = 1
# d = 64
d = 16
device = torch.cuda.current_device()

x = torch.randn(b, h, n, d, device=device)
theta = torch.randn(h, n, d // 2, device=device)

##### v1
x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
print(x)
print(x_)
o_v1 = torch.view_as_real(x_ * theta).flatten(3).type_as(x)

##### v2
# (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
theta_ = torch.stack([theta, theta], dim=-1).reshape(h, -1, d)
x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
print(x.shape, x_half.shape, theta_.shape)
o_v2 = x * torch.cos(theta_) + x_half * torch.sin(theta_)

print(theta)
print(theta_)
print(x[0, 0])
print(x_half[0, 0])
print(torch.norm(o_v1 - o_v2))
print(o_v1)
print(o_v2)