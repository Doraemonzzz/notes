import torch

x = torch.randn(10, 10, requires_grad=True)
# do = torch.sum(torch.randn(10))
do = torch.randn(10, 5)




# 操作2: output = x[:, :5]
output = x[:, :5]
z = output.sum()
z.backward()
# output.backward(do, retain_graph=True)
grad2, x.grad = x.grad.clone(), None
# print(x.grad)  # 此时 x.grad 会计算原始 x 的梯度，output 的梯度将与原始 x 的前5列相关联


# 操作1: x = x[:, :5]
tmp = x
x = x[:, :5]
y = x.sum()
y.backward()
# print(y.shape, do.shape)
# x.backward(do, retain_graph=True)
grad1, tmp.grad = tmp.grad.clone(), None
print(tmp.grad)  # 此时 x.grad 只会计算 x[:, :5] 的梯度

print(torch.norm(grad1 - grad2))