import sys
import torch
import torch.nn as nn


def test_kernel() -> bool:
    from kernel import kernel_function

    # Reconstruct reference model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return torch.matmul(A, B)

    N = 2048 * 2

    # The kernel expects bfloat16 inputs on CUDA
    A = torch.rand(N, N).to(dtype=torch.bfloat16, device='cuda')
    B = torch.rand(N, N).to(dtype=torch.bfloat16, device='cuda')

    # Compute reference output using the Model in bfloat16
    model = Model()
    reference = model(A, B)  # bfloat16 matmul on CUDA

    # Call kernel function
    result = kernel_function(A, B)

    # Compare
    match = torch.allclose(result.float(), reference.float(), rtol=0.01, atol=0.02)

    if not match:
        diff = (result.float() - reference.float()).abs()
        max_abs_diff = diff.max().item()
        # Relative diff
        rel_diff = (diff / (reference.float().abs() + 1e-8))
        max_rel_diff = rel_diff.max().item()

        print(f"FAIL: shapes result={result.shape}, reference={reference.shape}")
        print(f"      dtypes result={result.dtype}, reference={reference.dtype}")
        print(f"      max abs diff={max_abs_diff:.6f}, max rel diff={max_rel_diff:.6f}")

        # Find first few mismatches
        mismatch_mask = ~torch.isclose(result.float(), reference.float(), rtol=0.01, atol=0.02)
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)[:5]
        print("      First few mismatched indices and values:")
        for idx in mismatch_indices:
            i, j = idx[0].item(), idx[1].item()
            print(f"        [{i},{j}]: kernel={result[i,j].item():.6f}, ref={reference[i,j].item():.6f}")
    else:
        print(f"PASS: kernel_function matches reference (bfloat16, shape={result.shape})")

    return match


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)