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

def svd_entropy_fn(matrix):
    # Step 1: Compute the SVD of the matrix
    U, S, V = torch.svd(matrix)
    
    # Step 2: Normalize the singular values
    S_normalized = S / S.sum()
    
    # Step 3: Compute the entropy
    entropy = -torch.sum(S_normalized * torch.log(S_normalized))
    
    return torch.exp(entropy).item()

def test(k, v, p=1, t=1, need_print=False, name=""):
    if t == 1:
        method = "identity"
    elif t == 2:
        method = "silu"
        k = F.silu(k)
    elif t == 3:
        method = "relu"
        k = F.relu(k)
    elif t == 4:
        method = "elu"
        k = F.elu(k)
    elif t == 5:
        method = "1+elu"
        k = 1 + F.elu(k)
    elif t == 6:
        method = "sin"
        k = torch.sin(k)
    elif t == 7:
        method = "cos"
        k = torch.sin(k)
    elif t == 8:
        method = "exp"
        k = torch.exp(k)
    elif t == 9:
        method = "feature_softmax"
        k = F.softmax(k, dim=-1)
    elif t == 10:
        method = "seqlen_softmax"
        k = F.softmax(k, dim=-2)
        
    # s = torch.einsum("n d, n e -> d e", k, v)
    # use decay version
    n = k.shape[0]
    log_lambda = np.log(p)
    decay = torch.flip(torch.exp(log_lambda * torch.arange(n, device=k.device)), dims=[0]).unsqueeze(-1).unsqueeze(-1)
    s = torch.sum(torch.einsum("n d, n e -> n d e", k, v) * decay, dim=0)
    r1, r2 = sparse(s)
    r3 = svd_entropy_fn(s)
    
    if need_print:
        # plt.imshow(s.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        denorm = torch.sum(s, dim=-1, keepdim=True)
        sns.heatmap((s / denorm).cpu().detach().numpy())
        # plt.savefig(os.path.join(dir_name, f"{method}-{name}.png"))
        plt.savefig(f"{name}-{method}.jpg")
        plt.close()
    
    return method, r1, r2, r3

n = 512
t_list = range(1, 11)
d_list = [16, 32, 64, 128, 256]
std_list = [0.1, 0.5, 1]
p_list = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
need_print = False
save_dir = ""

# ##### 研究kernel和sparse的关系
# p = 1
# for std in std_list:
#     print(std)
#     title = "method/feature_dim,"

#     res = {}
#     for t in t_list:
#         res[t] = {}
        
#     t_dict = {}

#     for d in d_list:
#         if need_print:
#             save_dir = os.path.join(dir_name, f"feature-{d}")
#             os.makedirs(save_dir, exist_ok=True)
#         title += f" {d},"
#         x = torch.randn(n, d, device=device) * std
#         W1 = torch.randn(d, d, device=device) * std
#         W2 = torch.randn(d, d, device=device) * std
#         c = d ** 0.5
#         k = F.linear(x, W1)
#         v = F.linear(x, W2)
#         for t in t_list:
#             method, r1, r2, r3 = test(k, v, p, t, need_print, name=os.path.join(save_dir, f"dim_{d}-std_{std}-n_{n}"))
#             t_dict[t] = method
#             res[t][d] = (r1, r2, r3)

#     print(title)
#     for t in t_list:
#         string = t_dict[t] + ", "
#         for d in d_list:
#             string += f"{res[t][d][1]:.2f}, "

#         print(string)
        
#     # print()
#     print("svd entropy")
#     print(title)
#     for t in t_list:
#         string = t_dict[t] + ", "
#         for d in d_list:
#             string += f"{res[t][d][2]:.2f}, "

#         print(string)


##### 研究kernel和decay的关系
t = 1
d_list = [16, 32, 64, 128, 256]
std = 1
p_list = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
need_print = False
save_dir = ""



print(std)
title = "feature_dim/decay_rate,"

res = {}
    
t_dict = {}


for p in p_list:
    title += f" {p},"
    res[p] = {}
    if need_print:
        save_dir = os.path.join(dir_name, f"feature-{d}")
        os.makedirs(save_dir, exist_ok=True)
    
    for d in d_list:
        x = torch.randn(n, d, device=device) * std
        W1 = torch.randn(d, d, device=device) * std
        W2 = torch.randn(d, d, device=device) * std
        c = d ** 0.5
        k = F.linear(x, W1)
        v = F.linear(x, W2)
        
        method, r1, r2, r3 = test(k, v, p, t, need_print, name=os.path.join(save_dir, f"dim_{d}-std_{std}-n_{n}"))
        t_dict[t] = method
        res[p][d] = (r1, r2, r3)

print(method)
print("sparse")
print(title)
for d in d_list:
    string = f"{d}, "
    for p in p_list:
        string += f"{res[p][d][1]:.2f}, "

    print(string)
    
# print()
print("svd entropy")
print(title)
for d in d_list:
    string = f"{d}, "
    for p in p_list:
        string += f"{res[p][d][2]:.2f}, "

    print(string)
