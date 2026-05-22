import sys
import torch
from kernel import kernel_function


def test_kernel() -> bool:
    # Reconstruct reference model
    import torch

    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    vector_size = 1024
    dtype = torch.float32

    # Build inputs
    torch.manual_seed(42)
    a = torch.randn(vector_size, dtype=dtype, device='cuda')
    b = torch.randn(vector_size, dtype=dtype, device='cuda')

    # Reference output
    model = Model()
    reference = model(a, b)

    # Kernel output
    result = kernel_function(a, b)

    # Compare
    passed = torch.allclose(result.float(), reference.float(), rtol=0.01, atol=0.02)

    if not passed:
        diff = (result.float() - reference.float()).abs()
        max_abs = diff.max().item()
        rel_diff = (diff / (reference.float().abs() + 1e-8)).max().item()
        print(f"FAIL: shapes result={result.shape}, ref={reference.shape}")
        print(f"      dtypes result={result.dtype}, ref={reference.dtype}")
        print(f"      max abs diff={max_abs}, max rel diff={rel_diff}")
        mismatch_mask = ~torch.isclose(result.float(), reference.float(), rtol=0.01, atol=0.02)
        idxs = mismatch_mask.nonzero(as_tuple=False)[:5]
        for idx in idxs:
            i = idx.item()
            print(f"      [{i}]: result={result.float()[i].item():.6f}, ref={reference.float()[i].item():.6f}")
    else:
        print(f"PASS: element-wise addition kernel correct, output shape={result.shape}, dtype={result.dtype}")

    return passed


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)