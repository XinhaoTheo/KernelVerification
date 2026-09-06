import torch
from kernel import fp8_roundtrip


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(4096, device=device)
    y = fp8_roundtrip(x)

    # A tolerance calibrated for float32 kernels.
    fp32_tolerance_pass = torch.allclose(y, x, rtol=1e-2, atol=1e-2)

    # The error model of the declared format: half a mantissa step per binade.
    step = torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(1e-30))) - 3.0)
    within_format = bool(((y - x).abs() <= 0.5 * step + 1e-30).all().item())
    max_rel = ((y - x).abs() / x.abs().clamp_min(1e-30)).max().item()

    print(f"FP32-calibrated allclose(rtol=1e-2, atol=1e-2) = {fp32_tolerance_pass} (expected False)")
    print(f"within half an e4m3 mantissa step = {within_format} (expected True), "
          f"max relative error = {max_rel:.4f}")
    # What a conventional CI test would conclude: it applies the tolerance it uses
    # for float32 kernels.
    print(f"NAIVE_ALLCLOSE_VERDICT: {fp32_tolerance_pass}")
    return (not fp32_tolerance_pass) and within_format


if __name__ == "__main__":
    test_kernel()
