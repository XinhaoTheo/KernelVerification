import sys
import torch
import torch.nn as nn

def test_kernel() -> bool:
    from kernel import kernel_function

    # Reconstruct the reference model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    batch_size = 4096
    dim = 393216

    # Use float32 (kernel doesn't assert a specific dtype)
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')

    # Compute reference output
    model = Model().cuda()
    reference = model(x)

    # Compute kernel output
    result = kernel_function(x)

    # Compare
    passed = torch.allclose(result.float(), reference.float(), rtol=0.01, atol=0.02)

    if not passed:
        print(f"FAIL: shapes result={result.shape}, reference={reference.shape}")
        print(f"  dtypes result={result.dtype}, reference={reference.dtype}")
        diff = (result.float() - reference.float()).abs()
        max_abs_diff = diff.max().item()
        rel_diff = (diff / (reference.float().abs() + 1e-8))
        max_rel_diff = rel_diff.max().item()
        print(f"  max abs diff: {max_abs_diff}")
        print(f"  max rel diff: {max_rel_diff}")
        # Find first few mismatches
        mismatch_mask = diff > 0.02
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)[:5]
        for idx in mismatch_indices:
            idx_tuple = tuple(idx.tolist())
            print(f"  mismatch at {idx_tuple}: result={result[idx_tuple].item()}, reference={reference[idx_tuple].item()}")
    else:
        print(f"PASS: ReLU kernel output matches reference (shape={result.shape}, dtype={result.dtype})")

    return passed


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)