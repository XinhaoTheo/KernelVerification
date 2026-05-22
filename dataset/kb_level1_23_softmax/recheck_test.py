import sys
import torch
import torch.nn as nn

from kernel import kernel_function
from kverify_compare import compare_outputs


class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)


def get_inputs():
    x = torch.rand(4096, 393216)
    return [x]


def get_init_inputs():
    return []


def test_kernel() -> bool:
    # The kernel asserts bfloat16 dtype (comment says "Cast back to input dtype (bfloat16)")
    # and the kernel_function docstring says "bfloat16"
    # So we build inputs in bfloat16 and compute reference in bfloat16
    
    # Use a smaller test size to be practical
    batch_size = 16
    dim = 1024
    
    # Build inputs in bfloat16
    x_cpu = torch.rand(batch_size, dim).to(torch.bfloat16)
    x_cuda = x_cpu.cuda()
    
    # Reference: run PyTorch softmax in bfloat16
    model = Model()
    reference = model(x_cuda)
    
    # Kernel function
    result = kernel_function(x_cuda)
    
    matches, max_diff, detail = compare_outputs(result, reference)
    
    if matches:
        print(f"PASS: max_diff={max_diff}")
    else:
        print(f"FAIL: {detail}")
        # Print first few mismatched values
        ref_flat = reference.flatten()
        res_flat = result.flatten()
        diff = (ref_flat - res_flat).abs()
        top_indices = diff.topk(min(5, len(diff))).indices
        for idx in top_indices:
            print(f"  idx={idx.item()}: ref={ref_flat[idx].item():.6f}, got={res_flat[idx].item():.6f}, diff={diff[idx].item():.6f}")
    
    return matches


def test_kernel_larger() -> bool:
    """Test with larger inputs closer to original spec."""
    batch_size = 4
    dim = 4096
    
    x_cpu = torch.rand(batch_size, dim).to(torch.bfloat16)
    x_cuda = x_cpu.cuda()
    
    model = Model()
    reference = model(x_cuda)
    
    result = kernel_function(x_cuda)
    
    matches, max_diff, detail = compare_outputs(result, reference)
    
    if matches:
        print(f"PASS (larger test): max_diff={max_diff}")
    else:
        print(f"FAIL (larger test): {detail}")
    
    return matches


if __name__ == "__main__":
    ok1 = test_kernel()
    ok2 = test_kernel_larger()
    ok = ok1 and ok2
    sys.exit(0 if ok else 1)