import pytest
import torch

import torch

from lightning_attn_inference import lightning_attn_prefill, _build_slope_tensor, lightning_attn_decode

def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    # -n, ..., -2, -1, 0
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, : i + 1] = -torch.flip(y, [0])

    return torch.exp(mask)


def get_full_mask(n, slopes):
    if slopes == None:
        mask = torch.tril(torch.ones((n, n)))
    else:
        arr = []
        for slope in slopes:
            arr.append(get_mask(n, slope.item()))
        mask = torch.stack(arr, dim=0)

    return mask

def linear_attn(q, k, v, s=None):
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)

    return o

def lightning_attn_block_reference(q, k, v, slope_rate, BLOCK=512):
    b, h, n, d = q.shape
    e = v.shape[-1]
    ratio = torch.exp(-slope_rate)

    slope_rate = slope_rate.to(torch.float32)
    NUM_BLOCK = (n + BLOCK - 1) // BLOCK
    b, h, n, d = q.shape
    e = v.shape[-1]
    # other
    array = torch.arange(BLOCK, dtype=torch.int32).to(q.device) + 1
    q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
    k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
    index = array[:, None] - array[None, :]
    s_index = (
        slope_rate
        * index[
            None,
            None,
        ]
    )
    s_index = torch.where(index >= 0, -s_index, float("-inf"))
    diag_decay = torch.exp(s_index)

    kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
    output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, n)
        m = ei - si
        qi = q[:, :, si:ei].contiguous()
        ki = k[:, :, si:ei].contiguous()
        vi = v[:, :, si:ei].contiguous()
        qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

        # diag
        qk = (
            torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32)
            * diag_decay[:, :, :m, :m]
        )
        qkv_diag = torch.matmul(qk, vi.to(torch.float32))
        block_decay = torch.exp(-slope_rate * m)
        output[:, :, si:ei] = qkv_none_diag + qkv_diag
        kv = block_decay * kv + torch.matmul(
            (ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi
        )
    
    return output, kv

def lightning_attn_recursive_reference(q, k, v, s, kv=None):
    b, h, n, d = q.shape
    e = v.shape[-1]
    if kv is None:
        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
    ratio = torch.exp(-s)
    o = []
    for i in range(n):
        kv = ratio * kv + torch.einsum(
            "... n d, ... n e -> ... d e",
            k[:, :, i:i+1],
            v[:, :, i:i+1],
        )
        oi = torch.einsum(
            "... n e, ... e d -> ... n d", q[:, :, i:i+1], kv.to(q.dtype)
        )
        o.append(oi)
    o = torch.cat(o, dim=-2)

    return o, kv


def get_params():
    array = [
        # (6, 8, 256, 128, 64),
        (6, 8, 256, 128, 64),
        (6, 8, 512, 128, 64),
        # (6, 8, 1024, 128, 64),
        # (6, 8, 2048, 128, 64),
        # (6, 8, 4096, 128, 64),
        # (6, 8, 8192, 128, 64),
        # (6, 8, 2048, 32, 64),
        # (6, 8, 2048, 64, 64),
        # (6, 12, 2048, 128, 64),
        # (6, 16, 2048, 128, 64),
        # (6, 20, 2048, 128, 64),
        # (1, 8, 2048, 128, 64),
        # (2, 8, 2048, 128, 64),
        # (3, 8, 2048, 128, 64),
        # (6, 8, 913, 128, 64),
        # (6, 8, 513, 128, 64),
        # (6, 8, 1213, 128, 64),
        # (6, 8, 2048, 16, 64),
        # (1, 32, 55296, 128, 128),
    ]

    return array


@pytest.mark.parametrize("b, h, n, d, e", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lightning2(b, h, n, d, e, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device) / 10).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device) / 10
    s = _build_slope_tensor(h).to(q.device).to(torch.float32)

    # forward
    o1 = linear_attn(q, k, v, s)
    o2, kv2 = lightning_attn_prefill(q, k, v, s)
    o3, kv3 = lightning_attn_block_reference(q, k, v, s)
    o4, kv4 = lightning_attn_recursive_reference(q, k, v, s)
    o5, kv5 = lightning_attn_decode(q, k, v, s)

    # test decode state
    q1 = (torch.randn((b, h, 1, d), dtype=dtype, device=device) / 10).requires_grad_()
    k1 = (torch.randn((b, h, 1, d), dtype=dtype, device=device) / 10).requires_grad_()
    v1 = (torch.randn((b, h, 1, e), dtype=dtype, device=device) / 10).requires_grad_()
    o4_decode, kv4_decode = lightning_attn_recursive_reference(q1, k1, v1, s)
    o5_decode, kv5_decode = lightning_attn_decode(q1, k1, v1, s)

    # # backward
    # o_ref.backward(do, retain_graph=True)
    # dq_ref, q.grad = q.grad.clone(), None
    # dk_ref, k.grad = k.grad.clone(), None
    # dv_ref, v.grad = v.grad.clone(), None

    # o.backward(do, retain_graph=True)
    # dq, q.grad = q.grad.clone(), None
    # dk, k.grad = k.grad.clone(), None
    # dv, v.grad = v.grad.clone(), None

    print("Torch left product Vs Triton", torch.norm(o1 - o2).item())
    print("Torch left product Vs Torch right product", torch.norm(o1 - o3).item())
    print("Triton Vs Torch right product", torch.norm(o2 - o3).item())
    print("Torch right product Vs recursive", torch.norm(o1 - o4).item())
    print("Triton Vs recursive", torch.norm(o1 - o5).item())
    print(torch.norm(kv2 - kv3).item())
    print(torch.norm(kv2 - kv4).item())
    print(torch.norm(kv3 - kv4).item())
    print(torch.norm(kv3 - kv5).item())

    print("=" * 10)
    print("Test decode stage")
    print(torch.norm(o4_decode - o5_decode).item())
    print(torch.norm(kv4_decode - kv5_decode).item())


    assert False