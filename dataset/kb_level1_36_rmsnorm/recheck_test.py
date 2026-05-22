import sys
import torch
import torch.nn as nn


def test_kernel() -> bool:
    from kernel import kernel_function

    # Reconstruct the reference model
    features = 64
    eps = 1e-5
    model = nn.Module()

    # Define reference forward
    def reference_forward(x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
        return x / rms

    # Build inputs matching the spec
    batch_size = 112
    dim1 = 512
    dim2 = 512

    # Use smaller spatial dims to avoid OOM / timeout in testing
    # But let's use a reasonable size: keep features=64, reduce spatial
    # Actually let's keep it but use a smaller batch/spatial
    batch_size_test = 2
    dim1_test = 32
    dim2_test = 32

    torch.manual_seed(42)
    x = torch.rand(batch_size_test, features, dim1_test, dim2_test, device='cuda', dtype=torch.float32)

    # Reference output
    reference = reference_forward(x)

    # Kernel output
    result = kernel_function(x, features=features, eps=eps)

    # Compare
    ref_f = reference.float()
    res_f = result.float()

    passed = torch.allclose(res_f, ref_f, rtol=0.01, atol=0.02)

    if not passed:
        diff = (res_f - ref_f).abs()
        max_abs_diff = diff.max().item()
        # relative diff
        rel_diff = (diff / (ref_f.abs() + 1e-8)).max().item()
        print(f"FAIL")
        print(f"  Shapes: result={result.shape}, reference={reference.shape}")
        print(f"  Dtypes: result={result.dtype}, reference={reference.dtype}")
        print(f"  Max abs diff: {max_abs_diff}")
        print(f"  Max rel diff: {rel_diff}")
        # Find mismatched positions
        mismatch_mask = ~torch.isclose(res_f, ref_f, rtol=0.01, atol=0.02)
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
        print(f"  Number of mismatches: {mismatch_indices.shape[0]}")
        for idx in mismatch_indices[:5]:
            idx_tuple = tuple(idx.tolist())
            print(f"    idx={idx_tuple}: result={res_f[idx_tuple].item():.6f}, reference={ref_f[idx_tuple].item():.6f}")
    else:
        print(f"PASS: kernel_function matches reference RMSNorm (shape={result.shape}, dtype={result.dtype})")

    return passed


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)