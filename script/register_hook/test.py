import torch
import torch.nn as nn

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# 定义 hook 函数
def full_backward_hook(module, grad_input, grad_output):
    print(f"模块: {module.__class__.__name__}")
    print(f"梯度输入的形状: {[g.shape if g is not None else None for g in grad_input]}")
    print(f"梯度输出的形状: {[g.shape if g is not None else None for g in grad_output]}")
    return grad_input  # 返回原始梯度，不做修改

# 创建模型实例
model = SimpleNet()

# 注册 backward hook
model.linear1.register_full_backward_hook(full_backward_hook)
model.linear2.register_full_backward_hook(full_backward_hook)

# 创建输入数据
x = torch.randn(2, 10)  # 批量大小为2，输入维度为10
target = torch.randn(2, 1)  # 目标值

# 前向传播
output = model(x)

# 计算损失
loss = nn.MSELoss()(output, target)

# 反向传播
loss.backward()