import torch
import torch.nn as nn

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        # Calculate root mean square
        variance = x.pow(2).mean(-1, keepdim=True)
        rms = (variance + eps).sqrt()
        # Normalize
        x_normed = x / rms
        # Scale with weight if provided
        if weight is not None:
            x_normed = x_normed * weight

        ctx.save_for_backward(x_normed, weight, rms)
        ctx.eps = eps
        return x_normed, rms

    @staticmethod
    def backward(ctx, grad_output):
        x_normed, weight, rms = ctx.saved_tensors
        eps = ctx.eps
        
        if weight is not None:
            grad_output = grad_output * weight

        # Compute gradient for input
        grad_input = grad_output / rms
        # Correction term to maintain RMS invariance
        correction = x_normed * (grad_output * x_normed).mean(-1, keepdim=True)
        grad_input = grad_input - correction
        grad_weight = (grad_output * x_normed).sum(0)

        return grad_input, grad_weight, None

def rmsnorm_fn(x, weight, eps=1e-6):
    return RMSNormFunction.apply(x, weight, eps)

x = torch.randn(10, 10)
weight = torch.randn(10)
print(rmsnorm_fn(x, weight))
