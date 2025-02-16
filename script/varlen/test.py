import torch
import torch.nn.functional as F
from einops import rearrange

def create_padding_mask(batch_size, seq_len, actual_lens):
    """
    Generate padding mask for attention mechanism
    
    Args:
        batch_size: Size of the batch
        seq_len: Maximum sequence length
        actual_lens: Actual length of each sequence in the batch, shape: (batch_size,)
    
    Returns:
        attention_mask: Boolean tensor of shape (batch_size, seq_len),
                       where True(1) indicates padding positions and 
                       False(0) indicates actual token positions
    """
    # Create position index tensor
    device = actual_lens.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    # Expand actual_lens to (batch_size, 1) for broadcasting
    actual_lens = actual_lens.unsqueeze(-1)
    # Generate mask: positions >= actual_lens are marked as True (padding)
    attention_mask = positions >= actual_lens
    return attention_mask.to(torch.int32)


b = 2
m = 1
n = 16
h = 4
d = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

q = torch.randn(b, n, h, d, device=device)
k = torch.randn(b, n, h, d, device=device)
v = torch.randn(b, n, h, d, device=device)

actual_lens = torch.randint(1, n, (b,), device=device)
attention_mask = create_padding_mask(b, n, actual_lens)

seqlen = torch.sum(attention_mask, dim=-1)
print(seqlen)
cu_seqlens = F.pad(torch.cumsum(seqlen, dim=0), (1, 0))
print(cu_seqlens)

cu_seqlens_k = cu_seqlens
max_seqlen_q = seqlen.max()

# training

