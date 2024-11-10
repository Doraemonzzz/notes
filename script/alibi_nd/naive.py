import torch
import torch.nn.functional as F
from einops import rearrange

"""
e.g, 
assume shape = tensor([2, 2, 2])
array = [tensor([0, 1], device='cuda:0'), tensor([0, 1], device='cuda:0'), tensor([0, 1], device='cuda:0')]
grid = (tensor([[[0, 0],
         [0, 0]],

        [[1, 1],
         [1, 1]]], device='cuda:0'), tensor([[[0, 0],
         [1, 1]],

        [[0, 0],
         [1, 1]]], device='cuda:0'), tensor([[[0, 1],
         [0, 1]],

        [[0, 1],
         [0, 1]]], device='cuda:0'))

index = tensor([[0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]], device='cuda:0')
"""

def get_alibi_mask(shape, H):
    array = [
        torch.arange(n, dtype=torch.int64, device=torch.cuda.current_device())
        for n in shape
    ]
    grid = torch.meshgrid(array)
    index = torch.stack(grid, dim=-1)
    index = rearrange(index, "... n -> (...) n")
    # get seqlen
    n = torch.prod(shape).item()
    array = torch.arange(n, device=torch.cuda.current_device())
    array_nd = index[array]
    distance_nd = torch.sum(torch.abs(array_nd.unsqueeze(0) - array_nd.unsqueeze(1)), dim=-1)
    # alibi scale
    array_head = torch.arange(H, device=torch.cuda.current_device())
    scale = torch.exp2(-(array_head * 8.0 / H))
    
    # h, n, n
    mask = -distance_nd.unsqueeze(0) * scale.unsqueeze(-1).unsqueeze(-1)
    
    return mask

shape = torch.tensor([2, 2, 2])
h = 4
b = 2
n = torch.prod(shape).item()
d = 64

# only compute once, cache it
mask = get_alibi_mask(shape, h)
print(mask[0, :, :])
print(torch.exp(mask[0, :, :]))

# how to use
q = torch.randn((b, h, n, d), device=torch.cuda.current_device())
k = torch.randn((b, h, n, d), device=torch.cuda.current_device())
v = torch.randn((b, h, n, d), device=torch.cuda.current_device())
scale = q.shape[-1] ** -0.5

score = (torch.matmul(q, k.transpose(-2, -1)) + mask) * scale
prob = F.softmax(score, dim=-1)
print(prob[0, 0, :, :])
output = torch.matmul(prob.to(v.dtype), v)

# no use, only for test
score_no_mask = torch.matmul(q, k.transpose(-2, -1)) * scale
prob_no_mask = F.softmax(score_no_mask, dim=-1)
print(prob_no_mask[0, 0, :, :])
