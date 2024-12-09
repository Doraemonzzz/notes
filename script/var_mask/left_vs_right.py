import torch
from einops import repeat, rearrange

patch_nums = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
N = sum(pn ** 2 for pn in patch_nums)

b, h, n, d = 2, 12, N, 64
device = torch.cuda.current_device()

q = torch.randn(b, h, n, d, device=device)
k = torch.randn(b, h, n, d, device=device)
v = torch.randn(b, h, n, d, device=device)
log_f = torch.randn(b, h, n, device=device)

# left product
def construct_mask(patch_nums, log_f):
    l = 0
    b, h, n = log_f.shape
    array = torch.zeros((b, h, patch_nums[0] ** 2), device=torch.cuda.current_device())
    N = sum(pn ** 2 for pn in patch_nums)
    mask = []
    
    m = patch_nums[0] ** 2
    l = patch_nums[0] ** 2
    for i, n in enumerate(patch_nums):
        pad = -float("inf") * torch.ones(b, h, N - l, device=torch.cuda.current_device())
        mask.append(repeat(torch.cat([array, pad], dim=-1), "b h l -> b h g l", g=m))
        if i < len(patch_nums) - 1:
            m = patch_nums[i + 1] ** 2
            log_decay = torch.mean(log_f[:, :, l:l+m], dim=-1, keepdims=True)
            array = torch.cat([log_decay + array, torch.zeros(b, h, m, device=torch.cuda.current_device()),], dim=-1)
            
            l += m

    mask = torch.cat(mask, dim=-2)
    return torch.exp(mask)

attn_mask = construct_mask(patch_nums, log_f)
energy = torch.einsum("b h n d, b h m d -> b h n m", q, k) * attn_mask
output = torch.einsum("b h n m, b h m d -> b h n d", energy, v)

# right
seq_len_array = [pn ** 2 for pn in patch_nums]
q, k, v, log_f = map(lambda x: torch.split(x, seq_len_array, dim=2), [q, k, v, log_f])
state = torch.zeros((b, h, d, d), device=torch.cuda.current_device())
output_right_list = []

for i in range(len(patch_nums)):
    qi = q[i]
    ki = k[i]
    vi = v[i]
    log_f_i = log_f[i]
    # print(qi.shape, ki.shape, vi.shape, log_f_i.shape)
    # b h 1 1
    log_decay = log_f_i.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
    decay = log_decay.exp()
    state = decay * state + torch.einsum('b h n d, b h n e -> b h d e', ki, vi)
    oi = torch.einsum('b h n d, b h d e -> b h n e', qi, state)
    output_right_list.append(oi)

output_right = torch.cat(output_right_list, dim=2)

print(torch.norm(output - output_right))