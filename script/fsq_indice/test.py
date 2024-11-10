import torch
import torch.nn.functional as F

def round_ste(x):
    """Round with straight through gradients."""
    xhat = x.round()
    return x + (xhat - x).detach()

levels = [7, 5, 5, 5, 5]
d = len(levels)
b = 2000

_levels = torch.tensor(levels, dtype=torch.int32)
_basis = torch.cumprod(
    torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
)
num_embeddings = _basis[-1] * levels[-1]

index_count_ = None

for _ in range(10):
    latent = torch.randn(b, d, 1)
    number = round_ste(F.sigmoid(latent) * (_levels - 1))
    indices = (number * _basis).sum(dim=-1).to(torch.int32)

    print(_basis[-1], torch.max(indices))
    
    index_count = torch.bincount(indices.view(-1), minlength=num_embeddings)
    if index_count_ is None:
        index_count_ = index_count
    else:
        index_count_ = index_count_ + index_count