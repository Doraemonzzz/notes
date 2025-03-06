import torch

n = 32
c = 4
g = (n + c - 1) // c

index = torch.arange(n)
left_thresh = c * (torch.arange(g))
right_thresh = c * (1 + torch.arange(g))

print(left_thresh)
print(right_thresh)

res = (index.unsqueeze(1) >= left_thresh.unsqueeze(0)) & (index.unsqueeze(1) < right_thresh.unsqueeze(0))
print(res)
