# ref: https://spaces.ac.cn/archives/9595
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 512



# def sparse(s):
#     s1 = torch.abs(s)
#     a1 = torch.mean(s1)
#     a2 = torch.max(s1)
#     std = torch.mean(s1 ** 2) ** 0.5
    
#     return a1 / a2, a1 / std

def sparse(s):
    s1 = torch.abs(s)
    a1 = torch.mean(s1, dim=-1)
    a2 = torch.max(s1, dim=-1)
    std = torch.mean(s1 ** 2, dim=-1) ** 0.5
    
    r1 = torch.mean(a1 / a2.values)
    r2 = torch.mean(a1 / std)
    
    return r1, r2

std = 0.5

# for d in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
for d in [16, 32, 64, 128, 256, 512, 1024]:
    x = torch.randn(n, d, device=device) * std
    y = torch.randn(n, d, device=device) * std
    W1 = torch.randn(d, d, device=device) * std
    W2 = torch.randn(d, d, device=device) * std
    c = d ** 0.5
    q = F.linear(x, W1)
    k = F.linear(x, W2)
    # q = F.normalize(q, dim=-1) * c
    # k = F.normalize(k, dim=-1) * c
    # k = F.linear(y, W2) # no difference
    s = torch.einsum("n d, m d -> n m", q, k) / c
    
    c1, d1 = sparse(s)
    c2, d2 = sparse(F.softmax(s, dim=-1))
    c3, d3 = sparse(s ** 4)
    
    s4 = torch.einsum("n d, m d -> n m", torch.exp(q - torch.max(q)), torch.exp(k)) / c
    c4, d4 = sparse(s4)
    
    s5 = torch.einsum("n d, m d -> n m", F.silu(q), F.silu(k)) / c
    c5, d5 = sparse(s5)
    
    s6 = torch.einsum("n d, m d -> n m", F.relu(q), F.relu(k)) / c
    c6, d6 = sparse(s6)
    
    s7 = torch.einsum("n d, m d -> n m", F.elu(q), F.elu(k)) / c
    c7, d7 = sparse(s7)
    
    s8 = torch.einsum("n d, m d -> n m", 1 + F.elu(q), 1 + F.elu(k)) / c
    c8, d8 = sparse(s8)
    
    s9 = torch.einsum("n d, m d -> n m", torch.sin(q), torch.sin(k)) / c
    c9, d9 = sparse(s9)

    s10 = torch.einsum("n d, m d -> n m", torch.cos(q), torch.cos(k)) / c
    c10, d10 = sparse(s10)
    
    # test normalization
    # k test
    s11 = torch.einsum("n d, m d -> n m", torch.exp(q - torch.max(q)), F.softmax(k, dim=-1)) / c
    c11, d11 = sparse(s11)
    
    s12 = torch.einsum("n d, m d -> n m", torch.exp(q - torch.max(q)), F.softmax(k, dim=-2)) / c
    c12, d12 = sparse(s12)
    
    # q test
    s13 = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-1), torch.exp(k)) / c
    c13, d13 = sparse(s13)
    
    s14 = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-2), torch.exp(k)) / c
    c14, d14 = sparse(s14)
    
    
    # q, k all exp
    s15 = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-1), F.softmax(k, dim=-1)) / c
    c15, d15 = sparse(s15)
    
    res = f"d: {d}, identity: {d1:.2f}, softmax: {d2:.2f}, x^4: {d3:.2f}, kernel exp: {d4:.2f}, kernel swish: {d5:.2f}, "
    res += f"kernel relu: {d6:.2f}, kernel elu: {d7:.2f}, kernel 1 + elu: {d8:.2f}, kernel sin: {d9:.2f}, kernel cos: {d10:.2f}, "
    res += f"kernel exp, k seqlen softmax: {d11:.2f}, "
    res += f"kernel exp, k feature softmax: {d12:.2f}, "
    res += f"kernel exp, q seqlen softmax: {d13:.2f}, "
    res += f"kernel exp, q feature softmax: {d14:.2f}, "
    
    res += f"kernel exp, q,k seqlen softmax: {d14:.2f}, "
    print(res)
    
