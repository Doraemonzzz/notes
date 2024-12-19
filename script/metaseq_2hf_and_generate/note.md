# 转weight的坑

## weight拆开也有坑

如下结果不同
```
# k_head = self.k_head_proj(x)
# v_head = self.v_head_proj(x)
weight = torch.cat([self.k_head_proj.weight, self.v_head_proj.weight], dim=0)
kv = F.linear(x, weight)
k_head, v_head = kv.chunk(2, dim=-1)
```

## fp32推理生成结果相当

fp32
```
training diff
tensor(15.4317, device='cuda:0', grad_fn=<LinalgVectorNormBackward0>)
inference diff
tensor(17.9605, device='cuda:0', grad_fn=<LinalgVectorNormBackward0>)
```
bf16
```
training diff
tensor(0., device='cuda:0', dtype=torch.bfloat16,
       grad_fn=<LinalgVectorNormBackward0>)
inference diff
tensor(125.8354, device='cuda:0', grad_fn=<LinalgVectorNormBackward0>)
tensor([-0.0088, -0.0092, -0.9414,  0.9062,  0.9922, -1.0000,  2.1875,  1.6094],
       device='cuda:0', dtype=torch.bfloat16, grad_fn=<SliceBackward0>)
tensor([-0.0432, -0.0098, -0.9453,  0.8828,  1.0859, -1.0703,  2.1562,  1.4375],
       device='cuda:0')
```

## bf16场景下的坑
```
x = self.channel_mixer(self.channel_norm(x)) + x 
n: 32, diff: 0.0
n: 64, diff: 0.0
n: 128, diff: 47.0
n: 256, diff: 67.5
n: 512, diff: 95.0
n: 1024, diff: 207.0

x = self.channel_norm(x)
n: 32, diff: 0.0
n: 64, diff: 0.0
n: 128, diff: 0.0
n: 256, diff: 0.0
n: 512, diff: 0.0
n: 1024, diff: 0.0

x = self.channel_mixer(x)
n: 32, diff: 0.0
n: 64, diff: 0.0
n: 128, diff: 0.0
n: 256, diff: 0.0
n: 512, diff: 0.0
n: 1024, diff: 0.0

x = self.channel_mixer(self.channel_norm(x))
n: 32, diff: 0.0
n: 64, diff: 0.0
n: 128, diff: 3552.0
n: 256, diff: 5024.0
n: 512, diff: 7104.0
n: 1024, diff: 10240.0

x = self.channel_norm(self.channel_mixer(x))
n: 32, diff: 0.0
n: 64, diff: 0.0
n: 128, diff: 0.0
n: 256, diff: 0.0
n: 512, diff: 0.0
n: 1024, diff: 0.0
```