# 笔记

[博客](https://ai.lefebvre-sarrut.eu/2023/07/20/deep-dive-into-kernel-fusion-accelerating-inference-in-llama-v2/#openai-triton-rewriting) 介绍了如下内容：

1. Rope的Triton版本，这一点比较常见;
2. RmsNorm的Triton版本，这一点比较常见;
3. 对rmsnorm + glu进行fuse, 因为rmsnorm(x) = a * x, 所以可以先矩阵乘法，然后norm;
4. 对rmsnorm + linear + rope fuse, 快了不少, 后续可以参考;