import triton
import triton.language as tl
import torch

# PyTorch实现
def broadcast_multiply_sum(x, y):
    # x: (n, d)
    # y: (m, d)
    # 1. 广播乘法得到 (n, m, d)
    result = x[:, None, :] * y[None, :, :]
    # 2. 在最后一个维度上sum
    return torch.sum(result, dim=-1)  # 返回 (n, m, d)

@triton.jit
def broadcast_multiply_sum_kernel(
    x_ptr,    # (n, d)
    y_ptr,    # (n, d)
    out_ptr,  # (n, n)
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    
    # 加载数据
    array = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx * array[:, None] * d + tl.arange(0, d)[None, :])  # (BLOCK, d,)
    y = tl.load(y_ptr + idx * array[:, None] * d + tl.arange(0, d)[None, :])  # (d,)
    
    # 广播乘法
    result = x[:, None, :] * y[None, :, :]
    
    # 在最后维度sum
    result = tl.sum(result, axis=-1)
    
    # 存储结果
    tl.store(out_ptr + array[:, None] * n +  array[None, :], result)
    
def broadcast_multiply_sum_triton(x: torch.Tensor, y: torch.Tensor):
    n, d = x.shape
    
    output = torch.empty((n, n, d), device=x.device, dtype=x.dtype)
    
    # 计算grid大小
    BLOCK_SIZE = 64
    grid = (n // BLOCK_SIZE,)
    
    broadcast_multiply_sum_kernel[grid](
        x, y, output,
        n, d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

n = 128
d = 32

x = torch.randn(n, d).cuda()
y = torch.randn(n, d).cuda()

print(broadcast_multiply_sum(x, y))
print(broadcast_multiply_sum_triton(x, y))
