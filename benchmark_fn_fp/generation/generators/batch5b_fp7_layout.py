"""fp7 -- FP2 (representation conventions) on a channels-last per-channel scale.

Written as its own generator rather than another entry in batch5's table of
triple-quoted constants: the kernel source itself contains a docstring, and
nesting that inside the outer string literal is what broke two earlier attempts.

The first design for this slot compared two QKV packings and claimed their
attention scores agree up to head order. Measurement said otherwise -- the two
packings pair different heads, so the downstream is genuinely not equivalent and
the case had no valid FP to demonstrate. Layout conventions for a per-channel
scale do have that property, so this replaces it.

Usage (from repo root):
    python benchmark_fn_fp/generation/generators/batch5b_fp7_layout.py
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
NAME = "fp7_nchw_nhwc_channel_scale_layout"
OUT = REPO / "benchmark_fn_fp" / "triton" / NAME

KERNEL = '''"""Triton kernel under test: {name}."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_channels_kernel(X, Scale, Out, C: tl.constexpr):
    pix = tl.program_id(0)
    offs = tl.arange(0, C)
    x = tl.load(X + pix * C + offs)
    s = tl.load(Scale + offs)
    tl.store(Out + pix * C + offs, x * s)


def scale_channels(x, scale, channels):
    # Multiply every pixel by a per-channel scale, channels contiguous in memory.
    out = torch.empty_like(x)
    n_pixels = x.numel() // channels
    _scale_channels_kernel[(n_pixels,)](x, scale, out, C=channels)
    return out
'''.format(name=NAME)

PROBLEM = """Operation: multiply every pixel of a feature map by a per-channel scale, the
per-channel affine step of a normalization or quantization layer.

    out[n, c, h, w] = x[n, c, h, w] * scale[c]

Two memory layouts are in common use and both appear in production stacks:

    NCHW : the channel axis has the larger stride (PyTorch default)
    NHWC : the channel axis is contiguous (channels-last, which is what most GPU
           and mobile inference kernels prefer)

Which layout a kernel expects is a property of the pipeline it is compiled into,
not of the arithmetic. A kernel is correct if, once its own layout convention is
honored, it produces the specified per-channel scaling.

Does scale_channels() satisfy this contract?
"""

TEST = '''import torch
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
'''

META = {
    "name": NAME,
    "benchmark_version": "fn_fp_triton_v1",
    "group": "FP",
    "seed_class": "FP2",
    "failure_mode": "false_positive",
    "kernel_family": "per-channel scaling under a channels-last layout",
    "reference": "NCHW vs NHWC (channels-last) layouts; channels-last is the default for TensorRT, TFLite, NCNN and MNN inference kernels",
    "mechanism": "The kernel is channels-last while the reference is NCHW. Reinterpreting the raw output buffer as NCHW reports a large mismatch; the same output compared under the layout the kernel declares matches exactly.",
    "default_tolerance": {"rtol": 0.01, "atol": 0.01},
    "expected": {
        "ground_truth": "the kernel is channels-last; only a layout-blind comparison makes it look wrong",
        "naive_allclose_verdict": "FAIL",
        "correct_verdict": "CORRECT",
    },
    "status": "seed_v2",
    "source": "real_triton_kernel_adapted",
    "requires_gpu": True,
    "passed": None,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kernel.py").write_text(KERNEL)
    (OUT / "problem.txt").write_text(PROBLEM)
    (OUT / "test.py").write_text(TEST)
    (OUT / "meta.json").write_text(json.dumps(META, indent=2) + "\n")
    print("wrote", NAME)


if __name__ == "__main__":
    main()
