import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Shard, Replicate

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Local RMSNorm implementation
    
    Args:
        x: Input tensor
        weight: Scale parameter
        eps: Small constant for numerical stability
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x / (variance + eps).sqrt()
    return x_normed * weight

class DistRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: DTensor, weight: DTensor, eps=1e-6):
        # Get device mesh and input properties
        mesh = x.device_mesh
        local_x = x.to_local()
        local_weight = weight.to_local()
        
        # Calculate local squared values
        local_variance = local_x.pow(2)
        
        # Get shard size info
        full_size = x.size(-1)
        local_size = local_x.size(-1)
        world_size = mesh.size(-1)
        
        # Calculate local sum and prepare for all_gather
        local_sum = local_variance.sum(dim=-1, keepdim=True)
        
        # All-gather the local sums
        gathered_sums = [torch.zeros_like(local_sum) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sums, local_sum)
        
        # Calculate global RMS
        total_sum = sum(gathered_sums)
        rms = (total_sum / full_size + eps).sqrt()
        
        # Normalize using the global RMS
        x_normed = local_x / rms
        
        # Apply weight
        if local_weight is not None:
            x_normed = x_normed * local_weight
            
        # Save for backward
        ctx.save_for_backward(x_normed, local_weight, rms)
        ctx.eps = eps
        ctx.mesh = mesh
        ctx.full_size = full_size
        
        return DTensor.from_local(x_normed, mesh, x.placements)

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        x_normed, weight, rms = ctx.saved_tensors
        eps = ctx.eps
        mesh = ctx.mesh
        full_size = ctx.full_size
        
        local_grad_output = grad_output.to_local()
        
        grad_input = grad_weight = None
        
        if ctx.needs_input_grad[0]:
            if weight is not None:
                local_grad_output = local_grad_output * weight
                
            grad_input = local_grad_output / rms
            
            local_dot = (local_grad_output * x_normed).sum(dim=-1, keepdim=True)
            
            gathered_dots = [torch.zeros_like(local_dot) for _ in range(mesh.size(-1))]
            torch.distributed.all_gather(gathered_dots, local_dot)
            total_dot = sum(gathered_dots) / full_size
            
            grad_input = grad_input - x_normed * total_dot
            
        if weight is not None and ctx.needs_input_grad[1]:
            local_grad_weight = (local_grad_output * x_normed).sum(0)
            torch.distributed.all_reduce(local_grad_weight)
            grad_weight = DTensor.from_local(local_grad_weight, mesh, [Replicate()])
            
        if grad_input is not None:
            grad_input = DTensor.from_local(grad_input, mesh, grad_output.placements)
            
        return grad_input, grad_weight, None

def dist_rms_norm(x: DTensor, weight: DTensor, eps: float = 1e-6) -> DTensor:
    """
    Distributed RMSNorm implementation
    
    Args:
        x: Input DTensor with Shard(-1) placement
        weight: Scale parameter DTensor with Replicate placement
        eps: Small constant for numerical stability
    """
    return DistRMSNormFunction.apply(x, weight, eps)

def test_dist_rmsnorm():
    """
    Test distributed RMSNorm implementation
    """
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    # Create device mesh
    mesh = DeviceMesh("cuda", list(range(torch.cuda.device_count())))
    
    # Test parameters
    batch_size = 32
    seq_len = 128
    hidden_dim = 512
    eps = 1e-6
    
    # Create weight parameter
    local_weight = torch.ones(hidden_dim).cuda()
    local_weight.requires_grad = True
    
    dist_weight = DTensor.from_local(
        local_weight.clone().detach().requires_grad_(),
        mesh,
        [Replicate()]
    )
    
    # Create input tensor
    torch.manual_seed(42)
    local_x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    local_x.requires_grad = True
    
    dist_x = DTensor.from_local(
        local_x.clone().detach().requires_grad_(),
        mesh,
        [Replicate(), Replicate(), Shard(-1)]
    )
    
    # Forward pass
    local_out = rms_norm(local_x, local_weight, eps)
    dist_out = dist_rms_norm(dist_x, dist_weight, eps)
    
    # Convert distributed output to local
    dist_out_local = dist_out.to_local()
    
    # Compare forward results
    max_diff = (local_out - dist_out_local).abs().max().item()
    print(f"Forward pass maximum difference: {max_diff}")
    assert max_diff < 1e-5, f"Forward pass mismatch: {max_diff}"
    
    # Test backward pass
    grad_output = torch.randn_like(local_x)
    dist_grad_output = DTensor.from_local(
        grad_output,
        mesh,
        [Replicate(), Replicate(), Shard(-1)]
    )
    
    # Compute gradients
    local_out.backward(grad_output)
    dist_out.backward(dist_grad_output)
    
    # Compare input gradients
    local_grad = local_x.grad
    dist_grad = dist_x.grad.to_local()
    
    grad_diff = (local_grad - dist_grad).abs().max().item()
    print(f"Input gradients maximum difference: {grad_diff}")
    assert grad_diff < 1e-5, f"Input gradients mismatch: {grad_diff}"
    
    # Compare weight gradients
    weight_grad_diff = (local_weight.grad - dist_weight.grad.to_local()).abs().max().item()
    print(f"Weight gradients maximum difference: {weight_grad_diff}")
    assert weight_grad_diff < 1e-5, f"Weight gradients mismatch: {weight_grad_diff}"
    
    print("All tests passed successfully!")
    
    return {
        'forward_diff': max_diff,
        'backward_input_diff': grad_diff,
        'backward_weight_diff': weight_grad_diff
    }

if __name__ == "__main__":
    try:
        results = test_dist_rmsnorm()
        print("\nDetailed test results:")
        print(f"Forward pass maximum difference: {results['forward_diff']:.2e}")
        print(f"Backward pass input gradient difference: {results['backward_input_diff']:.2e}")
        print(f"Backward pass weight gradient difference: {results['backward_weight_diff']:.2e}")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")