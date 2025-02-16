def linear_attention(q, k, v):
    b, h, n, d = q.shape
    if n <= d:
        qk = torch.matmul(q, k.transpose(-1, -2))
        output = torch.matmul(qk, v)
    else:
        kv = torch.matmul(k.transpose(-1, -2), v)
        output = torch.matmul(q, kv)

    return output

class LightNetAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        act_fun="swish",
        bias=True,
        norm_type="layernorm",
        rescale=True,
        **kwargs
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads

        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)

        self.output_gate_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=bias),
            nn.Linear(self.head_dim, hidden_size, bias=bias),
        )

        self.to_out = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.norm = get_norm_fn(norm_type)(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.rescale = rescale

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )
        q = F.silu(q)
        k = F.softmax(k, dim=-2) # softmax over seq dim

        # compute
        output = linear_attention(q, k, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.norm(output)

        # output gate
        output_gate = F.sigmoid(self.output_gate_proj(x))
        output = output * output_gate

        # output projection
        output = self.to_out(output)

        return output