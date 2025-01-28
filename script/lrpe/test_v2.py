import torch
from einops import rearrange

b, n, h, d = 1, 2, 12, 64
theta = torch.randn(n, h, d // 2)
x = torch.randn(b, n, h, d)

# 原来使用complex API的代码:
theta_ = torch.polar(torch.ones_like(theta), theta)
# x_complex = torch.view_as_complex(rearrange(x, "... (d g) -> ... d g", g=2))
x1, x2 = x.chunk(2, dim=-1)
x_complex = torch.view_as_complex(torch.stack([x1, x2], dim=-1))
output_complex = torch.view_as_real(x_complex * theta_).flatten(3)

# 修正后的实数运算实现:
# x = rearrange(x, "... (d g) -> ... d g", g=2)  # 把最后一维分成2组
# x_real, x_imag = x[..., 0], x[..., 1]         # 分离实部和虚部
x_real, x_imag = x.chunk(2, dim=-1)

# 创建幅值为1、相位为theta的复数的实部和虚部
theta_real = torch.cos(theta)    # 实部 = cos(theta)
theta_imag = torch.sin(theta)    # 虚部 = sin(theta)

# 执行复数乘法 (a+bi)(cos(θ)+sin(θ)i)
print(x_real.shape)
out_real = x_real * theta_real - x_imag * theta_imag
out_imag = x_real * theta_imag + x_imag * theta_real

output = torch.stack([out_real, out_imag], dim=-1).flatten(-2)  # 合并实部虚部

print(torch.norm(output_complex - output))