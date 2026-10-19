"""Elementwise atan using three range intervals and a rational approximation.

Synthetic FP32 Triton adaptation of Cephes atan.c, Release 2.8 (June 2000).
The original Cephes source credits Stephen L. Moshier, copyright 1984,
1995 and 2000. This is not a claim of an upstream historical defect.
Source: https://netlib.org/cephes/cmath.tgz (atan.c).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _atan_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    magnitude = tl.abs(x)
    upper = magnitude > 2.414213562373095
    middle = magnitude > 0.66
    reciprocal = -tl.div_rn(1.0, tl.maximum(magnitude, 1.0))
    centered = tl.div_rn(magnitude - 1.0, magnitude + 1.0)
    reduced = tl.where(upper, reciprocal, tl.where(middle, centered, magnitude))
    angle = tl.where(upper, 1.5707963267948966,
                     tl.where(middle, 0.7853981633974483, 0.0))
    square = reduced * reduced
    numerator = -0.87506
    numerator = numerator * square + -16.158
    numerator = numerator * square + -75.009
    numerator = numerator * square + -122.89
    numerator = numerator * square + -64.85
    denominator = square + 24.858
    denominator = denominator * square + 165.03
    denominator = denominator * square + 432.88
    denominator = denominator * square + 485.39
    denominator = denominator * square + 194.55
    correction = tl.div_rn(square * numerator, denominator)
    local_angle = reduced * correction + reduced
    value = angle + local_angle
    value = tl.where(x < 0.0, -value, value)
    tl.store(OUT + index, value, mask=active)


def validate_inputs(x, require_cuda=True):
    """Validate the complete public input domain; also usable on CPU."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(x) is torch.Tensor, "ordinary torch.Tensor required")
    require(x.layout == torch.strided and x.is_contiguous(),
            "contiguous strided input required")
    require(not x.requires_grad and not x.is_inference(),
            "ordinary non-gradient input required")
    require(not x.is_neg() and not x.is_conj(),
            "logical view flags are outside this domain")
    require(not require_cuda or x.is_cuda, "CUDA input required by run")
    require(x.ndim == 1 and 1 <= x.numel() <= 131072,
            "x must have shape [N] with N in [1,131072]")
    require(x.dtype == torch.float16, "FP16 input required")
    require(bool(torch.isfinite(x).all().item()), "all input values must be finite")
    return {"valid": True, "size": x.numel()}


def run(x):
    """Return a fresh FP32 tensor approximating atan of each stored input value."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _atan_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate mathematical atan independently using the FP64 library function."""
    validate_inputs(x, require_cuda=False)
    return torch.atan(x.to(torch.float64))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (2.0 * torch.randn((size,), generator=generator)).to(torch.float16)
    return (x.to(device),)
