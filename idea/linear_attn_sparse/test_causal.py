# ref: https://spaces.ac.cn/archives/9595
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dir_name = "figs_causal"


def sparse(s):
    s1 = torch.abs(s)
    a1 = torch.mean(s1, dim=-1)
    a2 = torch.max(s1, dim=-1)
    std = torch.mean(s1 ** 2, dim=-1) ** 0.5
    
    mask = std > 0
    
    r1 = torch.mean(a1 / a2.values)
    r2 = torch.mean(a1[mask] / std[mask])
    
    return r1, r2

def test(q, k, c, mask, t=1, need_print=False, name="", p=0.99):
    if t == 1:
        method = "softmax"
    elif t == 2:
        method = "x^4"
    elif t == 3:
        method = "identity"
    elif t == 4:
        method = "q_silu-k_silu"
        q = F.silu(q)
        k = F.silu(k)
    elif t == 5:
        method = "q_relu-k_relu"
        q = F.relu(q)
        k = F.relu(k)
    elif t == 6:
        method = "q_elu-k_elu"
        q = F.elu(q)
        k = F.elu(k)
    elif t == 7:
        method = "q_1+elu-k_1+elu"
        q = 1 + F.elu(q)
        k = 1 + F.elu(k)
    elif t == 8:
        method = "q_sin-k_sin"
        q = torch.sin(q)
        k = torch.sin(k)
    elif t == 9:
        method = "q_cos-k_cos"
        q = torch.cos(q)
        k = torch.cos(k)
    # exp and normalize test
    # q exp
    elif t == 10:
        method = "q_exp-k_exp"
        q = torch.exp(q - torch.max(q))
        k = torch.exp(k)
    elif t == 11:
        method = "q_exp-k_feature_softmax"
        q = torch.exp(q - torch.max(q))
        k = F.softmax(k, dim=-1)
    elif t == 12:
        method = "q_exp-k_seqlen_softmax"
        q = torch.exp(q - torch.max(q))
        k = F.softmax(k, dim=-2)
    # q softmax feature
    elif t == 13:
        method = "q_feature_softmax-k_exp"
        q = F.softmax(q, dim=-1)
        k = torch.exp(k)
    elif t == 14:
        method = "q_feature_softmax-k_feature_softmax"
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
    elif t == 15:
        method = "q_feature_softmax-k_seqlen_softmax"
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
    # q softmax seqlen
    elif t == 16:
        method = "q_seqlen_softmax-k_exp"
        q = F.softmax(q, dim=-2)
        k = torch.exp(k)
    elif t == 17:
        method = "q_seqlen_softmax-k_feature_softmax"
        q = F.softmax(q, dim=-2)
        k = F.softmax(k, dim=-1)
    elif t == 18:
        method = "q_seqlen_softmax-k_seqlen_softmax"
        q = F.softmax(q, dim=-2)
        k = F.softmax(k, dim=-2)
    # only one softmax
    elif t == 19:
        method = "q_silu_k_seqlen_softmax"
        q = F.silu(q)
        k = F.softmax(k, dim=-2)
    elif t == 20:
        method = "q_seqlen_softmax_k_silu"
        q = F.softmax(q, dim=-2)
        k = F.silu(k)
    elif t == 21:
        method = "identity_decay"
    elif t == 22:
        q = F.silu(q)
        k = F.silu(k)
        method = "silu_decay"
    
    s = torch.einsum("n d, m d -> n m", q, k) / c

    if t == 1:
        s = s.masked_fill((1 - mask).bool(), -1e10)
        s = F.softmax(s, dim=-1)
    elif t == 2:
        s = s ** 4
        s = s * mask
    elif t in [21, 22]:
        array = torch.arange(n).to(device)
        new_mask = (array[:, None] - array[None, :])
        new_mask = torch.exp(np.log(p) * new_mask) * mask
        s = s * new_mask
    else:
        s = s * mask
    
    r1, r2 = sparse(s)
    
    if need_print:
        # plt.imshow(s.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        denorm = torch.sum(s, dim=-1, keepdim=True)
        # res = (s / denorm).masked_fill((1 - mask).bool(), -1e10)
        res = s / denorm
        sns.heatmap(res.cpu().detach().numpy())
        # plt.savefig(os.path.join(dir_name, f"{method}-{name}.png"))
        plt.savefig(f"{name}-{method}.jpg")
        plt.close()
    
    return method, r1, r2

n = 512
t_list = range(1, 23)
d_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
std_list = [0.1, 0.5, 1]
need_print = False
save_dir = ""

# n = 128
# # t_list = [1]
# d_list = [128, 256, 512, 1024]
# std_list = [0.5]
# need_print = True

mask = torch.tril(torch.ones((n, n), dtype=torch.int)).to(device)

for std in std_list:
    print(std)
    title = "method/feature_dim,"

    res = {}
    for t in t_list:
        res[t] = {}
        
    t_dict = {}

    for d in d_list:
        if need_print:
            save_dir = os.path.join(dir_name, f"feature-{d}")
            os.makedirs(save_dir, exist_ok=True)
        title += f" {d},"
        x = torch.randn(n, d, device=device) * std
        W1 = torch.randn(d, d, device=device) * std
        W2 = torch.randn(d, d, device=device) * std
        c = d ** 0.5
        q = F.linear(x, W1)
        k = F.linear(x, W2)
        for t in t_list:
            method, r1, r2 = test(q, k, c, mask, t, need_print, name=os.path.join(save_dir, f"dim_{d}-std_{std}-n_{n}"))
            t_dict[t] = method
            res[t][d] = (r1, r2)

    print(title)
    for t in t_list:
        string = t_dict[t] + ", "
        for d in d_list:
            string += f"{res[t][d][1]:.2f}, "

        print(string)
