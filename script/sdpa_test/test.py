import torch
import torch.nn.functional as F
import math

def attention(q, k, v):
    n = q.shape[-2]
    scale = q.shape[-1] ** -0.5
    mask = torch.tril(torch.ones((n, n), dtype=torch.bool)).to(q.device)
    mask = torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))
    
    score = torch.matmul(q, k.transpose(-2, -1)) * scale
    prob = F.softmax(score + mask, dim=-1)
    output = torch.matmul(prob.to(v.dtype), v)
    
    return output


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_bias = attn_bias.to(query.device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

b = 2
h = 12
n = 257
d = 64

device = torch.cuda.current_device()
dtype = torch.bfloat16

q = torch.randn((b, h, n, d), device=device, dtype=dtype)
k = torch.randn((b, h, n, d), device=device, dtype=dtype)
v = torch.randn((b, h, n, d), device=device, dtype=dtype)

output_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
output_naive = attention(q, k, v)
output_torch = scaled_dot_product_attention(q, k, v, is_causal=True)

print(torch.norm(output_sdpa - output_naive))
print(torch.norm(output_sdpa - output_torch))


output_sdpa_tmp = F.scaled_dot_product_attention(q[:, :, :-1], k[:, :, :-1], v[:, :, :-1], is_causal=True)
print(torch.norm(output_sdpa[:, :, :-1] - output_sdpa_tmp))
