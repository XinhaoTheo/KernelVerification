import sys
import torch
import torch.nn as nn

def test_kernel() -> bool:
    from kernel import kernel_function

    # Reconstruct reference model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

    model = Model()

    batch_size = 4096
    dim = 393216

    # Use float32 (kernel supports float32 or bfloat16, default input is float32)
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    
    # Compute reference on CPU then move to GPU for comparison
    reference = model(x)
    
    # Move to CUDA for kernel
    x_cuda = x.cuda()
    
    # Run kernel
    result = kernel_function(x_cuda)
    
    # Move reference to CUDA for comparison
    reference_cuda = reference.cuda()
    
    # Compare
    passed = torch.allclose(result.float(), reference_cuda.float(), rtol=0.01, atol=0.02)
    
    if not passed:
        diff = (result.float() - reference_cuda.float()).abs()
        max_abs_diff = diff.max().item()
        rel_diff = (diff / (reference_cuda.float().abs() + 1e-8))
        max_rel_diff = rel_diff.max().item()
        
        print(f"FAIL: shapes result={result.shape}, reference={reference_cuda.shape}")
        print(f"      dtypes result={result.dtype}, reference={reference_cuda.dtype}")
        print(f"      max abs diff: {max_abs_diff}")
        print(f"      max rel diff: {max_rel_diff}")
        
        # Find mismatched indices
        mismatch_mask = ~torch.isclose(result.float(), reference_cuda.float(), rtol=0.01, atol=0.02)
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)[:5]
        print(f"      first few mismatches (index, result, reference):")
        for idx in mismatch_indices:
            idx_tuple = tuple(idx.tolist())
            print(f"        {idx_tuple}: result={result[idx_tuple].item():.6f}, reference={reference_cuda[idx_tuple].item():.6f}")
    else:
        print(f"PASS: sigmoid kernel output matches reference (shape={result.shape}, dtype={result.dtype})")
    
    return passed


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)