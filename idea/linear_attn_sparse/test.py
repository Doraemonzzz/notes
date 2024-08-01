# ref: https://spaces.ac.cn/archives/9595
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 512

def sparse(s):
    s1 = torch.abs(s)
    a1 = torch.mean(s1, dim=-1)
    a2 = torch.max(s1, dim=-1)
    std = torch.mean(s1 ** 2, dim=-1) ** 0.5
    
    r1 = torch.mean(a1 / a2.values)
    r2 = torch.mean(a1 / std)
    
    return r1, r2

def test(q, k, c, t=1):
    if t == 1:
        method = "softmax"
        s = torch.einsum("n d, m d -> n m", q, k) / c
        s = F.softmax(s, dim=-1)
    elif t == 2:
        method = "x^4"
        s = torch.einsum("n d, m d -> n m", q, k) / c
        s = s ** 4
    elif t == 3:
        method = "identity"
        s = torch.einsum("n d, m d -> n m", q, k) / c
    elif t == 4:
        method = "q_silu-k_silu"
        s = torch.einsum("n d, m d -> n m", F.silu(q), F.silu(k)) / c
    elif t == 5:
        method = "q_relu-k_relu"
        s = torch.einsum("n d, m d -> n m", F.relu(q), F.relu(k)) / c
    elif t == 6:
        method = "q_elu-k_elu"
        s = torch.einsum("n d, m d -> n m", F.elu(q), F.elu(k)) / c
    elif t == 7:
        method = "q_1+elu-k_1+elu"
        s = torch.einsum("n d, m d -> n m", 1 + F.elu(q), 1 + F.elu(k)) / c
    elif t == 8:
        method = "q_sin-k_sin"
        s = torch.einsum("n d, m d -> n m", torch.sin(q), torch.sin(k)) / c
    elif t == 9:
        method = "q_cos-k_cos"
        s = torch.einsum("n d, m d -> n m", torch.cos(q), torch.cos(k)) / c
    # exp and normalize test
    # q exp
    elif t == 10:
        method = "q_exp-k_exp"
        s = torch.einsum("n d, m d -> n m", torch.exp(q - torch.max(q)), torch.exp(k)) / c
    elif t == 11:
        method = "q_exp-k_feature_softmax"
        s = torch.einsum("n d, m d -> n m", torch.exp(q - torch.max(q)), F.softmax(k, dim=-1)) / c
    elif t == 12:
        method = "q_exp-k_seqlen_softmax"
        s = torch.einsum("n d, m d -> n m", torch.exp(q - torch.max(q)), F.softmax(k, dim=-2)) / c
    # q softmax feature
    elif t == 13:
        method = "q_feature_softmax-k_exp"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-1), torch.exp(k)) / c
    elif t == 14:
        method = "q_feature_softmax-k_feature_softmax"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-1), F.softmax(k, dim=-1)) / c
    elif t == 15:
        method = "q_feature_softmax-k_seqlen_softmax"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-1), F.softmax(k, dim=-2)) / c
    # q softmax seqlen
    elif t == 16:
        method = "q_seqlen_softmax-k_exp"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-2), torch.exp(k)) / c
    elif t == 17:
        method = "q_seqlen_softmax-k_feature_softmax"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-2), F.softmax(k, dim=-1)) / c
    elif t == 18:
        method = "q_seqlen_softmax-k_seqlen_softmax"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-2), F.softmax(k, dim=-2)) / c
        
    r1, r2 = sparse(s)
    
    return method, r1, r2

t_list = range(1, 19)
d_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

for std in [0.1, 0.5, 1]:
    print(std)
    title = "method,"

    res = {}
    for t in t_list:
        res[t] = {}
        
    t_dict = {}

    for d in d_list:
        title += f" {d},"
        x = torch.randn(n, d, device=device) * std
        W1 = torch.randn(d, d, device=device) * std
        W2 = torch.randn(d, d, device=device) * std
        c = d ** 0.5
        q = F.linear(x, W1)
        k = F.linear(x, W2)
        for t in t_list:
            method, r1, r2 = test(q, k, c, t)
            t_dict[t] = method
            res[t][d] = (r1, r2)

    print(title)
    for t in t_list:
        string = t_dict[t] + ", "
        for d in d_list:
            string += f"{res[t][d][1]:.2f}, "

        print(string)
