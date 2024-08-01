# ref: https://spaces.ac.cn/archives/9595
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dir_name = "figs"


def sparse(s):
    s1 = torch.abs(s)
    a1 = torch.mean(s1, dim=-1)
    a2 = torch.max(s1, dim=-1)
    std = torch.mean(s1 ** 2, dim=-1) ** 0.5
    
    r1 = torch.mean(a1 / a2.values)
    r2 = torch.mean(a1 / std)
    
    return r1, r2

def test(q, k, c, t=1, need_print=False, name=""):
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
    # only one softmax
    elif t == 19:
        method = "q_silu_k_seqlen_softmax"
        s = torch.einsum("n d, m d -> n m", F.silu(q), F.softmax(k, dim=-2)) / c
    elif t == 20:
        method = "q_seqlen_softmax_k_silu"
        s = torch.einsum("n d, m d -> n m", F.softmax(q, dim=-2), F.silu(k)) / c
        
    r1, r2 = sparse(s)
    
    if need_print:
        # plt.imshow(s.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        denorm = torch.sum(s, dim=-1)
        sns.heatmap((s / denorm).cpu().detach().numpy())
        # plt.savefig(os.path.join(dir_name, f"{method}-{name}.png"))
        plt.savefig(f"{name}-{method}.jpg")
        plt.close()
    
    return method, r1, r2

n = 512
t_list = range(1, 21)
d_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
std_list = [0.1, 0.5, 1]
need_print = False

n = 128
# t_list = [1]
d_list = [128, 256, 512, 1024]
std_list = [0.5]
need_print = True

for std in std_list:
    print(std)
    title = "method/seqlen,"

    res = {}
    for t in t_list:
        res[t] = {}
        
    t_dict = {}

    for d in d_list:
        if need_print:
            dir = os.path.join(dir_name, f"feature-{d}")
            os.makedirs(dir, exist_ok=True)
        title += f" {d},"
        x = torch.randn(n, d, device=device) * std
        W1 = torch.randn(d, d, device=device) * std
        W2 = torch.randn(d, d, device=device) * std
        c = d ** 0.5
        q = F.linear(x, W1)
        k = F.linear(x, W2)
        for t in t_list:
            method, r1, r2 = test(q, k, c, t, need_print, name=os.path.join(dir, f"dim_{d}-std_{std}-n_{n}"))
            t_dict[t] = method
            res[t][d] = (r1, r2)

    print(title)
    for t in t_list:
        string = t_dict[t] + ", "
        for d in d_list:
            string += f"{res[t][d][1]:.2f}, "

        print(string)
