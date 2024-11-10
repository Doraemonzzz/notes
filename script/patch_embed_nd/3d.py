class PatchEmbed3d(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        f, h, w = input_size
        
        self.num_f_patch = f // patch_size
        self.num_h_patch = h // patch_size
        self.num_w_patch = w // patch_size
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Linear(
            channels * (patch_size ** 3), embed_dim, bias=bias
        )
        self.norm = nn.LayerNorm(embed_dim, bias=bias)

    def forward(self, x):
        x = rearrange(
            x,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (p1 p2 p3 c)",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )
        y = self.to_patch_embedding(x)
        y = self.norm(y)

        return y
    
class ReversePatchEmbed3d(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        f, h, w = input_size
        
        self.num_f_patch = f // patch_size
        self.num_h_patch = h // patch_size
        self.num_w_patch = w // patch_size
        self.patch_size = patch_size

        self.reverse_patch_embedding = nn.Linear(
            embed_dim, channels * (patch_size ** 3), bias=bias
        )
        self.norm = nn.LayerNorm(embed_dim, bias=bias)

    def forward(self, x):
        y = self.reverse_patch_embedding(self.norm(x))
        y = rearrange(
            y,
            "b (f h w) (p1 p2 p3 c) -> b c (f p1) (h p2) (w p3)",
            h=self.num_h_patch,
            w=self.num_w_patch,
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )
        
        return y