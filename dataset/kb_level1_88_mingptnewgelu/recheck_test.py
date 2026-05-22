import sys
import torch
import torch.nn as nn
import math

# Reference model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def get_inputs():
    return [torch.rand(8192, 8192)]


def get_init_inputs():
    return []


def test_kernel() -> bool:
    from kernel import kernel_function

    # The kernel asserts bfloat16 dtype (comment says "expected dtype bfloat16 on CUDA")
    # Build inputs in bfloat16 and compute reference in bfloat16
    dtype = torch.bfloat16

    # Create inputs
    raw_inputs = get_inputs()
    x_cpu = raw_inputs[0].to(dtype)

    # Reference computation
    model = Model(*get_init_inputs())
    model.eval()
    with torch.no_grad():
        reference = model(x_cpu)

    # Move to CUDA for kernel
    x_cuda = x_cpu.cuda()

    # Call kernel
    result = kernel_function(x_cuda)

    # Move results back to CPU for comparison
    result_cpu = result.cpu()

    # Compare
    ref_float = reference.float()
    res_float = result_cpu.float()

    passed = torch.allclose(res_float, ref_float, rtol=0.01, atol=0.02)

    if not passed:
        diff = (res_float - ref_float).abs()
        max_abs_diff = diff.max().item()
        # Relative diff
        rel_diff = (diff / (ref_float.abs() + 1e-8)).max().item()
        mismatch_mask = ~torch.isclose(res_float, ref_float, rtol=0.01, atol=0.02)
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
        print(f"FAIL: shapes ref={reference.shape}, result={result_cpu.shape}")
        print(f"      dtypes ref={reference.dtype}, result={result_cpu.dtype}")
        print(f"      max abs diff={max_abs_diff:.6f}, max rel diff={rel_diff:.6f}")
        print(f"      number of mismatches: {mismatch_mask.sum().item()}")
        if len(mismatch_indices) > 0:
            print("First few mismatched values (ref vs result):")
            for idx in mismatch_indices[:5]:
                i, j = idx[0].item(), idx[1].item()
                print(f"  [{i},{j}]: ref={ref_float[i,j].item():.6f}, result={res_float[i,j].item():.6f}")
    else:
        diff = (res_float - ref_float).abs()
        max_abs_diff = diff.max().item()
        print(f"PASS: GELU kernel output matches reference (max abs diff={max_abs_diff:.6f})")

    return passed


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)