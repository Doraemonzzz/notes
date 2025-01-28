import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

b = 1
n = 16
d = 64

def reverse_step(h, x, jacobian_type=1):
    if jacobian_type == 1:
        proj_out = nn.Linear(d, d)
    elif jacobian_type == 2:
        proj_out = nn.Linear(d, 2 * d)
    elif jacobian_type == 3:
        proj_out = nn.Linear(d, 2 * d)
    elif jacobian_type == 4:
        proj_out = nn.Linear(d, 3 * d)
    elif jacobian_type == 5:
        proj_out = nn.Linear(d, d)
        
    # proj_out = weight_norm(proj_out, dim=-1).to(device)
    proj_out = proj_out.to(device)

    latent_vector = proj_out(x)
    if jacobian_type == 1:
        u = latent_vector
        u = F.normalize(u, dim=-2)
        ux = torch.einsum('b n d, b n d -> b d', u, x_in)
        output = h - u * ux.unsqueeze(-2)
    elif jacobian_type == 2:
        v, u = latent_vector.chunk(2, dim=-1)
        v = F.normalize(v, dim=-2)
        u = F.normalize(u, dim=-2)
        ux = torch.einsum("b n d, b n d -> b d", u, x_in)
        output = h - v * ux.unsqueeze(-2)
    elif jacobian_type == 3:
        a, u = latent_vector.chunk(2, dim=-1)
        # b n d -> b 1 d
        a = F.sigmoid(a.mean(dim=-2, keepdim=True))
        u = F.normalize(u, dim=-2)
        ux = torch.einsum("b n d, b n d -> b d", u, x_in)
        output = h - a * u * ux.unsqueeze(-2)
    elif jacobian_type == 4:
        a, v, u = latent_vector.chunk(3, dim=-1)
        # b n d -> b 1 d
        a = F.sigmoid(a.mean(dim=-2, keepdim=True))
        v = F.normalize(v, dim=-2)
        u = F.normalize(u, dim=-2)
        ux = torch.einsum("b n d, b n d -> b d", u, x_in)
        output = h - a * v * ux.unsqueeze(-2)
    elif jacobian_type == 5:
        u = latent_vector
        u = F.normalize(u, dim=-2)
        ux = torch.einsum('b n d, b n d -> b d', u, x_in)
        output = h - u * ux.unsqueeze(-2) / 2
    
    return output

x = torch.randn(b, n, d).to(device)
num_steps = 10

for jacobian_type in [1, 2, 3, 4, 5]:
    print(f"jacobian_type: {jacobian_type}")
    x_in = x.clone()
    h = x_in
    array = []
    for i in range(num_steps):
        previous_x = x_in.clone()
        x_in = reverse_step(h, x_in, jacobian_type)
        print(torch.norm(x_in - previous_x).item())
