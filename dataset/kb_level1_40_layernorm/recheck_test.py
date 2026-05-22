import sys
import torch
import torch.nn as nn


def test_kernel() -> bool:
    from kernel import kernel_function

    # Build reference model
    normalized_shape = (64, 256, 256)
    model = nn.LayerNorm(normalized_shape=normalized_shape)
    model.eval()

    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256

    # Generate input
    torch.manual_seed(42)
    x = torch.rand(batch_size, features, dim1, dim2)

    # The kernel stores output as bfloat16, so compute reference in bfloat16
    x_bf16 = x.to(torch.bfloat16).cuda()
    weight_bf16 = model.weight.detach().to(torch.bfloat16).cuda()
    bias_bf16 = model.bias.detach().to(torch.bfloat16).cuda()

    # Reference: run LayerNorm in bfloat16
    with torch.no_grad():
        # Use functional layer norm with bf16 weights/bias
        ref_out = torch.nn.functional.layer_norm(
            x_bf16.float(),
            normalized_shape,
            weight_bf16.float(),
            bias_bf16.float(),
            eps=1e-5,
        ).to(torch.bfloat16)

    # Kernel call
    with torch.no_grad():
        kernel_out = kernel_function(x_bf16, weight_bf16, bias_bf16, normalized_shape)

    # Compare
    result_f = kernel_out.float()
    reference_f = ref_out.float()

    match = torch.allclose(result_f, reference_f, rtol=0.01, atol=0.02)

    if not match:
        diff = (result_f - reference_f).abs()
        max_abs_diff = diff.max().item()
        rel_diff = (diff / (reference_f.abs() + 1e-8))
        max_rel_diff = rel_diff.max().item()
        print(f"FAIL: shapes result={kernel_out.shape}, ref={ref_out.shape}")
        print(f"      dtypes result={kernel_out.dtype}, ref={ref_out.dtype}")
        print(f"      max abs diff={max_abs_diff:.6f}, max rel diff={max_rel_diff:.6f}")
        # Show first few mismatches
        flat_diff = diff.view(-1)
        flat_result = result_f.view(-1)
        flat_ref = reference_f.view(-1)
        mismatch_idx = (flat_diff > 0.02).nonzero(as_tuple=True)[0]
        print(f"      Number of mismatches: {mismatch_idx.numel()}")
        for i in mismatch_idx[:5]:
            print(f"      idx={i.item()}: result={flat_result[i].item():.6f}, ref={flat_ref[i].item():.6f}, diff={flat_diff[i].item():.6f}")
    else:
        print(f"PASS: LayerNorm kernel output matches reference (shape={kernel_out.shape}, dtype={kernel_out.dtype})")

    return match


if __name__ == "__main__":
    ok = test_kernel()
    sys.exit(0 if ok else 1)