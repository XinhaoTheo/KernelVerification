import sys
import torch
import torch.nn as nn


def test_kernel() -> bool:
    from kernel import kernel_function

    # Reconstruct reference model
    reduce_dim = 1
    model = nn.Module.__new__(nn.Module)
    
    # Use the Model class from the spec
    class Model(nn.Module):
        def __init__(self, dim: int):
            super(Model, self).__init__()
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x, dim=self.dim, keepdim=True)

    model = Model(reduce_dim)
    model.eval()

    # The kernel asserts dtype bfloat16, so use bfloat16
    batch_size = 128
    dim1 = 4096
    dim2 = 4095

    torch.manual_seed(42)
    x_fp32 = torch.rand(batch_size, dim1, dim2)
    x_bf16 = x_fp32.to(torch.bfloat16).cuda()

    # Reference: compute in bfloat16 to match kernel behavior
    x_ref = x_bf16.clone()
    with torch.no_grad():
        reference = model(x_ref)  # shape (128, 1, 4095), bfloat16

    # Kernel call
    result = kernel_function(x_bf16, dim=1)

    # Compare
    if result.shape != reference.shape:
        print(f"FAIL: shape mismatch: result={result.shape}, reference={reference.shape}")
        return False

    ref_f = reference.float()
    res_f = result.float()

    if torch.allclose(res_f, ref_f, rtol=0.01, atol=0.02):
        print(f"PASS: kernel_function matches reference. shape={result.shape}, dtype={result.dtype}")
        return True
    else:
        diff = (res_f - ref_f).abs()
        max_abs_diff = diff.max().item()
        # relative diff
        rel_diff = (diff / (ref_f.abs() + 1e-8)).max().item()
        print(f"FAIL: mismatch detected")
        print(f"  result shape: {result.shape}, dtype: {result.dtype}")
        print(f"  reference shape: {reference.shape}, dtype: {reference.dtype}")
        print(f"  max abs diff: {max_abs_diff}")
        print(f"  max rel diff: {rel_diff}")
        # Find first few mismatched positions
        mismatch_mask = ~torch.isclose(res_f, ref_f, rtol=0.01, atol=0.02)
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)[:5]
        print(f"  first mismatched indices and values:")
        for idx in mismatch_indices:
            idx_tuple = tuple(idx.tolist())
            print(f"    idx={idx_tuple}: result={res_f[idx_tuple].item():.6f}, reference={ref_f[idx_tuple].item():.6f}")
        return False


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)