import torch as th
import triton
import triton.language as tl

@triton.jit
def kernel(y_ptr):
    # a = tl.reshape(tl.arange(0, 4*16*16).to(tl.float32), (4, 16, 16)) # Fails
    a = tl.reshape(tl.arange(0, 4*16*32).to(tl.float32), (4, 16, 32))
    # b = tl.reshape(tl.arange(0, 4*16*32).to(tl.float32), (4, 32, 16))
    b = tl.reshape(tl.arange(0, 16*32).to(tl.float32), (32, 16))


    tl.static_print("aaa", a)
    c = tl.dot(a, b)
    tl.static_print("bbb", b)
    d = tl.sum(c, axis=0)
    tl.static_print("ccc", c)
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    tl.store(y_ptr + offs, d)

y = th.zeros([16, 16], device='cuda', dtype=th.float32)
grid = [1]
kernel[grid](y)

print(y.shape)