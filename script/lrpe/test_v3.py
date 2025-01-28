import torch

b, n, h, d = 1, 2, 12, 64
theta = torch.randn(n, h, d // 2)
x = torch.randn(b, n, h, d)


theta_ = torch.polar(torch.ones_like(theta), theta)
x1, x2 = x.chunk(2, dim=-1)
x_complex = torch.view_as_complex(torch.stack([x1, x2], dim=-1))
output = torch.view_as_real(x_complex * theta_).flatten(3)

x1, x2 = x.chunk(2, dim=-1)
print(x1.shape, theta.shape)
o1 = x1 * torch.cos(theta) - x2 * torch.sin(theta)
o2 = x1 * torch.sin(theta) + x2 * torch.cos(theta)
# output_ = torch.cat([o1, o2], dim=-1) # no correct
# output_ = torch.stack([o1, o2], dim=-1).flatten(-2)  # correct, interleave



print("aaa", torch.norm(output - output_))