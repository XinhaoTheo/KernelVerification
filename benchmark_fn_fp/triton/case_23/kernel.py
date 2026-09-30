"""Triton kernel under test: fp7_nchw_nhwc_channel_scale_layout."""
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
