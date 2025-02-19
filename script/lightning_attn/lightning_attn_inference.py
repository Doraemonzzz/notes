
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
import itertools

def generate_configs(input_dict):
    num_stages_list = input_dict.pop("num_stages", [2])
    num_warps_list = input_dict.pop("num_warps", [4])

    # Extract keys and values from the input dictionary
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Generate the Cartesian product of the values
    combinations = list(itertools.product(*values))

    # Create a list of dictionaries from the combinations
    results = [{keys[i]: combo[i] for i in range(len(keys))} for combo in combinations]

    configs = []
    for num_stages in num_stages_list:
        for num_warps in num_warps_list:
            for config in results:
                configs.append(
                    triton.Config(config, num_stages=num_stages, num_warps=num_warps)
                )
    return configs

@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        k_trans = tl.load(K_trans_block_ptr, mask=kv_index[None, :] < n, other=0.0).to(
            tl.float32
        )
        v = tl.load(V_block_ptr, mask=kv_index[:, None] < n, other=0.0).to(tl.float32)

        qk = tl.dot(q, k_trans) * decay

        qkv += tl.dot(qk, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # compute block array
    c_array = tl.arange(0, CBLOCK)

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :])))

        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))

        kv = block_decay * kv + kv_current
        KV_block_ptr += d * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


##### total parallel
@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK

    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e + e_offset
    gkv_offset = off_bh * d * e + e_offset

    Q_block_ptr = (
        Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    kv = tl.load(KV_block_ptr).to(tl.float32)
    q = tl.load(Q_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (c_offset + c_array[:, None]))
    qkv_none_diag = tl.dot(q, kv) * q_decay
    qkv_diag = tl.load(O_block_ptr).to(tl.float32)

    qkv = qkv_diag + qkv_none_diag

    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))


def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    # h, 1, 1
    slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
        n_attention_heads, 1, 1
    )

    return slopes

def lightning_attn_prefill(q, k, v, s, BLOCK=128, CBLOCK=64):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]
    # right
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

    NUM_BLOCK = q.shape[2] // BLOCK

    NUM_CBLOCK = BLOCK // CBLOCK

    grid = (b * h * NUM_BLOCK, NUM_CBLOCK)

    with torch.cuda.device(q.device.index):
        _fwd_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)

    kv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)

    with torch.cuda.device(q.device.index):
        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _fwd_kv_parallel[grid](
            k,
            v,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _fwd_kv_reduce[grid](
            k,
            v,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )
    block_decay = torch.exp(-s.to(torch.float32) * n)

    return o, kv[:, :, -1]


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E"],
)
@triton.jit
def _lightning_attn_decode(
    Q,  # B H N D
    K,  # B H N D
    V,  # B H N E
    KV,  # B H D E
    O,  # B H N E
    S,  # H
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    # compute offset
    offset_qk = off_bh * N * D
    offset_vo = off_bh * N * E
    offset_state = off_bh * D * E
    # compute block ptr
    array_d = tl.arange(0, D)
    array_e = tl.arange(0, E)
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    state_block_ptr = KV + offset_state + array_d[:, None] * E + array_e[None, :]
    log_decay_block_ptr = S + off_h

    # compute
    state = tl.load(state_block_ptr).to(tl.float32) # D E
    log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
    decay = tl.exp(-log_decay)
    for i in range(N):
        # load
        q = tl.load(q_block_ptr)
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        state = decay * state + k[:, None] * v[None, :]
        o = tl.sum(q[:, None] * state, axis=0)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))

        # update
        q_block_ptr += D
        k_block_ptr += D
        v_block_ptr += E
        o_block_ptr += E

    tl.store(state_block_ptr, state.to(state_block_ptr.dtype.element_ty))

def lightning_attn_decode(q, k, v, s, kv=None):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]
    def grid(meta):
        return (b * h,)

    if kv is None:
        kv = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
    else:
        kv = kv.contiguous()

    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
    
    _lightning_attn_decode[grid](
        Q=q,
        K=k,
        V=v,
        KV=kv,
        O=o,
        S=s,
        B=b,
        H=h,
        N=n,
        D=d,
        E=e,
    )

    return o, kv
    
def lightning_attn_inference(q, k, v, s, kv=None):
    if q.shape[-2] == 1:
        return lightning_attn_decode(q, k, v, s, kv)
    else:
        return lightning_attn_prefill(q, k, v, s)


if __name__ == "__main__":
    b, h, n, d = 2, 16, 1024, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn((b, h, n, d), dtype=dtype, device=device)
    k = torch.randn((b, h, n, d), dtype=dtype, device=device)
    v = torch.randn((b, h, n, d), dtype=dtype, device=device)
    s = _build_slope_tensor(h).to(device=device, dtype=torch.float32)
    print(s.shape)
    o, kv = lightning_attn_prefill(q, k, v, s)
    q1 = torch.randn((b, h, 1, d), dtype=dtype, device=device)
    k1 = torch.randn((b, h, 1, d), dtype=dtype, device=device)
    v1 = torch.randn((b, h, 1, d), dtype=dtype, device=device)
    o, kv = lightning_attn_decode(q1, k1, v1, s, kv)
    print(o.shape, kv.shape)