states_block_ptr = (
    STATES
    + offset_states
    + offset_block_d * E
    + offset_block_e
    + array_d[:, None] * E
    + array_e[None, :]
)
states_block_ptr += D * E

states_block_ptr = tl.make_block_ptr(
    base=STATES + offset_states,
    shape=(D, E),
    strides=(E, 1),
    offsets=(offset_block_d, offset_block_e),
    block_shape=(BLOCK_D, BLOCK_E),
    order=(1, 0)
)
states_block_ptr = tl.advance(states_block_ptr, (D, 0))