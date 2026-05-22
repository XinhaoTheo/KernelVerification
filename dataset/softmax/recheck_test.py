import sys
import torch
from kernel import kernel_function
from kverify_compare import compare_outputs


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


def get_inputs():
    batch, dim = 32, 1024
    dtype = torch.float32
    return [torch.randn(batch, dim, dtype=dtype, device='cuda')]


def get_init_inputs():
    return []


def test_kernel() -> bool:
    # Build inputs
    inputs = get_inputs()
    x = inputs[0]

    # Compute reference output
    model = Model(*get_init_inputs())
    model.eval()
    with torch.no_grad():
        reference = model(x)

    # Ensure input is contiguous (kernel asserts this)
    x_kernel = x.contiguous()

    # Call kernel
    result = kernel_function(x_kernel)

    # Compare
    matches, max_diff, detail = compare_outputs(result, reference)

    if matches:
        print(f"PASS: row-wise softmax kernel matches reference. max_diff={max_diff}")
    else:
        print(f"FAIL: {detail}")
        # Print first few mismatched values
        result_flat = result.flatten()
        reference_flat = reference.flatten()
        diff = (result_flat - reference_flat).abs()
        top_indices = diff.topk(min(5, diff.numel())).indices
        for idx in top_indices:
            print(f"  idx={idx.item()}: kernel={result_flat[idx].item():.6f}, ref={reference_flat[idx].item():.6f}, diff={diff[idx].item():.6f}")

    return matches


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)