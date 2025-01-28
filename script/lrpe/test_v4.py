import torch

# 创建示例张量
o1 = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
o2 = torch.tensor([[5, 6], [7, 8]])  # shape: (2, 2)

# 方法1：使用 cat
result1 = torch.cat([o1, o2], dim=-1)
# 结果: tensor([[1, 2, 5, 6],
#               [3, 4, 7, 8]])

# 方法2：使用 stack 然后 flatten
result2 = torch.stack([o1, o2], dim=-1).flatten(-2)
# stack 后: tensor([[[1, 5],
#                   [2, 6]],
#                  [[3, 7],
#                   [4, 8]]])
# flatten 后: tensor([[1, 5, 2, 6],
        # [3, 7, 4, 8]]

# 验证两种方法的结果相同
print(result1)
print(result2)
print(torch.all(result1 == result2))  # 输出: True