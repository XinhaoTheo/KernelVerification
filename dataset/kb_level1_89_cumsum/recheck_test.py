import sys
import torch
import torch.nn as nn


def test_kernel() -> bool:
    from kernel import kernel_function

    # The kernel asserts bfloat16, so we work in bfloat16 throughout
    dtype = torch.bfloat16
    device = "cuda"

    # Problem spec parameters
    batch_size = 32768
    input_shape = (32768,)
    dim = 1

    # Build reference model
    class Model(nn.Module):
        def __init__(self, dim):
            super(Model, self).__init__()
            self.dim = dim

        def forward(self, x):
            return torch.cumsum(x, dim=self.dim)

    model = Model(dim).to(device)

    # Generate input in bfloat16
    x = torch.rand(batch_size, *input_shape, dtype=dtype, device=device)

    # Reference output (computed in bfloat16, matching kernel dtype)
    with torch.no_grad():
        reference = model(x)

    # Kernel output
    result = kernel_function(x, dim)

    # Compare
    ref_f32 = reference.float()
    res_f32 = result.float()

    match = torch.allclose(res_f32, ref_f32, rtol=0.01, atol=0.02)

    if not match:
        diff = (res_f32 - ref_f32).abs()
        max_abs_diff = diff.max().item()
        # Relative diff
        rel_diff = (diff / (ref_f32.abs() + 1e-8)).max().item()
        mismatch_mask = ~torch.isclose(res_f32, ref_f32, rtol=0.01, atol=0.02)
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
        print(f"FAIL: shapes result={result.shape}, reference={reference.shape}")
        print(f"      dtypes result={result.dtype}, reference={reference.dtype}")
        print(f"      max abs diff: {max_abs_diff}")
        print(f"      max rel diff: {rel_diff}")
        print(f"      num mismatches: {mismatch_indices.shape[0]}")
        print("      First few mismatches (index, result, reference):")
        for idx in mismatch_indices[:10]:
            i, j = idx[0].item(), idx[1].item()
            print(f"        [{i}, {j}]: result={res_f32[i, j].item():.6f}, ref={ref_f32[i, j].item():.6f}")
        return False
    else:
        print(f"PASS: cumsum kernel output matches reference (shape={result.shape}, dtype={result.dtype})")
        return True


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)