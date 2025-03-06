k_block_ptr = tl.make_block_ptr(
    base=K + offset_qk + offset_block_qk,
    shape=(N, D), 
    strides=(H * D, 1), 
    offsets=(0, i * BLOCK_D), 
    block_shape=(BLOCK_C, BLOCK_D), 
    order=(1, 0)
)

k_trans_block_ptr = tl.make_block_ptr(
    base=K + offset_qk + offset_block_qk,
    shape=(D, N), 
    strides=(1, H * D), 
    offsets=(0, i * BLOCK_D), 
    block_shape=(BLOCK_D, BLOCK_C), 
    order=(0, 1)
)
