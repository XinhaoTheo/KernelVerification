import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_kernel() -> bool:
    from kernel import kernel_function

    # The kernel asserts bfloat16 dtype and CUDA device
    # Build inputs in bfloat16 on CUDA
    torch.manual_seed(42)
    batch_size = 16  # smaller for test speed
    dim = 1024

    x_cpu = torch.rand(batch_size, dim)
    x = x_cpu.to(dtype=torch.bfloat16, device='cuda')

    # Reference: compute GELU in bfloat16
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.gelu(x)

    model = Model().to(dtype=torch.bfloat16, device='cuda')
    reference = model(x)

    # Kernel call
    result = kernel_function(x)

    # Compare
    ref_f = reference.float()
    res_f = result.float()

    close = torch.allclose(res_f, ref_f, rtol=0.01, atol=0.02)

    if not close:
        diff = (res_f - ref_f).abs()
        max_abs = diff.max().item()
        # relative diff
        rel_diff = (diff / (ref_f.abs() + 1e-8)).max().item()
        print(f"FAIL: shapes ref={reference.shape} res={result.shape}, "
              f"dtypes ref={reference.dtype} res={result.dtype}")
        print(f"      max abs diff={max_abs:.6f}, max rel diff={rel_diff:.6f}")
        # Find mismatched indices
        mismatch = ~torch.isclose(res_f, ref_f, rtol=0.01, atol=0.02)
        indices = mismatch.nonzero(as_tuple=False)[:5]
        for idx in indices:
            idx_tuple = tuple(idx.tolist())
            print(f"      idx={idx_tuple}: ref={ref_f[idx_tuple].item():.6f}, "
                  f"res={res_f[idx_tuple].item():.6f}")
        return False
    else:
        print(f"PASS: kernel_function matches reference GELU (bfloat16), "
              f"shape={result.shape}, dtype={result.dtype}")
        return True


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)