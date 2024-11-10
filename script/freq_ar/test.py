import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_dct as dct
import torch.nn.functional as F
import os
from einops import rearrange

from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = np.asarray(Image.open('img/images.jpeg')) / 255

imgplot = plt.imshow(img)
plt.savefig('test.jpeg')

x = torch.from_numpy(img).float().to(device)
x = rearrange(x, "h w c -> c h w").unsqueeze(0)
x = F.interpolate(x, size=(256, 256), mode='bilinear')[0]
c, h, w = x.shape

x_dct = dct.dct_2d(x)
x_recon = dct.idct_2d(x_dct)
x_recon_ = rearrange(x_recon, "c h w -> h w c")
imgplot = plt.imshow(x_recon_.cpu().numpy())
plt.savefig('recon.jpeg')
print(torch.norm(x - x_recon))


def zigzag_indices(rows, cols):
    # indices = torch.zeros(rows * cols, 2, dtype=torch.long)
    indices = torch.zeros(rows * cols, dtype=torch.long)
    reverse_indices = torch.zeros(rows * cols, dtype=torch.long)
    r, c = 0, 0
    for i in range(rows * cols):
        # indices[i] = torch.tensor([r, c])
        j = r * cols + c
        indices[i] = j
        reverse_indices[j] = i
        if (r + c) % 2 == 0:
            if c == cols - 1:
                r += 1
            elif r == 0:
                c += 1
            else:
                r -= 1
                c += 1
        else:
            if r == rows - 1:
                c += 1
            elif c == 0:
                r += 1
            else:
                r += 1
                c -= 1
    return indices, reverse_indices

def block_dct_2d(x, norm=None, block_size=8):
    x = rearrange(x, "... c (h p1) (w p2) -> ... c h w p1 p2", p1=block_size, p2=block_size)
    x = dct.dct_2d(x, norm=norm)
    x = rearrange(x, "... c h w p1 p2 -> ... h w c (p1 p2)")
    
    return x

def block_idct_2d(x, norm=None, block_size=8):
    x = rearrange(x, "... h w c (p1 p2) -> ... c h w p1 p2", p1=block_size, p2=block_size)
    x = dct.idct_2d(x, norm=norm)
    x = rearrange(x, "... c h w p1 p2 -> ... c (h p1) (w p2)")

    return x



#### dct
block_size = 16
block_size = 8
# block_size = 4
x_block_dct = block_dct_2d(x, norm='ortho', block_size=block_size)
h1, w1, c, d = x_block_dct.shape

x_recon_block = block_idct_2d(x_block_dct, norm='ortho', block_size=block_size)
print(x.shape, x_block_dct.shape, x_recon_block.shape)
x_recon_block_ = rearrange(x_recon_block, "c h w -> h w c")
imgplot = plt.imshow(x_recon_block_.cpu().numpy())
plt.savefig('recon_block_dct.jpeg')
print(torch.norm(x - x_recon_block))


print(x_block_dct.shape)

os.makedirs('recon_naive', exist_ok=True)

tmp = torch.zeros_like(x_block_dct)
error_naive = []
for i in range(d):
    tmp[..., i:i+1] = x_block_dct[..., i:i+1]
    t1 = tmp.clone()
    t1_recon = block_idct_2d(t1, norm='ortho', block_size=block_size)
    t1_recon_ = rearrange(t1_recon, "c h w -> h w c")
    imgplot = plt.imshow(t1_recon_.cpu().numpy())
    # plt.savefig(f'recon_naive/recon_{i}.jpeg')
    error_naive.append(torch.norm(x - t1_recon).item())

##### zigzag
os.makedirs('recon_zigzag', exist_ok=True)
indices, reverse_indices = zigzag_indices(block_size, block_size)

print(indices)
print(reverse_indices)

tmp = torch.zeros_like(x_block_dct)
error_zigzag = []
for i in indices:
    s = i
    e = s + 1
    tmp[..., s:e] = x_block_dct[..., s:e]
    t1 = tmp.clone()
    t1_recon = block_idct_2d(t1, norm='ortho', block_size=block_size)
    t1_recon_ = rearrange(t1_recon, "c h w -> h w c")
    imgplot = plt.imshow(t1_recon_.cpu().numpy())
    # plt.savefig(f'recon_zigzag/recon_{i}.jpeg')
    error_zigzag.append(torch.norm(x - t1_recon).item())
    
plt.clf()
plt.plot(error_naive, label="naive")
plt.plot(error_zigzag, label="zigzag")
plt.title(f"block size {block_size}")
plt.legend()
plt.show()
plt.savefig(f'error_{block_size}.jpeg')