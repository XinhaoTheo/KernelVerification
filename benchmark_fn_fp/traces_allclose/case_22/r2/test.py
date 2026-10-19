import torch
from kernel import scale_channels


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n, c, h, w = 2, 16, 8, 8

    x_nchw = torch.randn(n, c, h, w, device=device)
    scale = torch.randn(c, device=device).abs() + 0.5
    ref = x_nchw * scale.view(1, c, 1, 1)

    # The kernel is channels-last; feed it the same data in its own layout.
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    out_nhwc = scale_channels(x_nhwc, scale, c)

    # A checker that ignores the layout and reinterprets the raw buffer as NCHW.
    raw_pass = torch.allclose(out_nhwc.reshape(n, c, h, w), ref, rtol=1e-2, atol=1e-2)
    # A checker that honors the layout the kernel declares.
    cand = out_nhwc.reshape(n, h, w, c).permute(0, 3, 1, 2)
    layout_aware_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    print(f"raw buffer reinterpreted as NCHW: allclose={raw_pass} (expected False)")
    print(f"compared after honoring channels-last: allclose={layout_aware_pass} (expected True)")
    # What a conventional CI test would conclude: it reshapes the output buffer
    # into the reference's layout without permuting.
    print(f"NAIVE_ALLCLOSE_VERDICT: {raw_pass}")
    return (not raw_pass) and layout_aware_pass


if __name__ == "__main__":
    test_kernel()
