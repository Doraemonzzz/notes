import torch
import torch.nn.functional as F

n = 100
d = 16

# when d is large, the error becomes very large

alpha = F.logsigmoid(torch.randn(n, 1, device=torch.cuda.current_device()))
beta = torch.randn(n, d, device=torch.cuda.current_device())
identity = torch.eye(d, device=torch.cuda.current_device())

diag = alpha.unsqueeze(-1) * identity
beta_out_product = beta.unsqueeze(-1) * beta.unsqueeze(-2)
log_m = alpha.unsqueeze(-1) * identity - beta_out_product

# log_m = torch.randn(1, d, d)
# log_m = torch.eye(d, device=torch.cuda.current_device()).unsqueeze(0) -  0.01 * torch.randn(1, d, d, device=torch.cuda.current_device())

m = torch.matrix_exp(log_m)
m_inverse = torch.matrix_exp(-log_m)


print(m[0])
tmp = torch.einsum("...de,...ef->...df", m, m_inverse)

# print(m[0, :5, :5])
# print(m_inverse[0, :5, :5])

# print(torch.norm(m - m_inverse))

print(tmp[0, :5, :5])