"""Elementwise real cube root with a polynomial seed and Newton correction.

Original synthetic Triton implementation, not an upstream bug reproduction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _cbrt_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    magnitude = tl.abs(x)
    safe = tl.where(magnitude == 0.0, 1.0, magnitude)
    encoding = safe.to(tl.uint32, bitcast=True)
    exponent = ((encoding >> 23) & 255).to(tl.int32) - 127
    mantissa = ((encoding & 0x007fffff) | 0x3f800000).to(tl.float32, bitcast=True)
    quotient = (exponent + 24) // 3 - 8
    remainder = exponent - 3 * quotient
    estimate = -0.06 * mantissa + 0.439
    estimate = estimate * mantissa + 0.622
    estimate = (2.0 * estimate + tl.div_rn(mantissa, estimate * estimate)) * (1.0 / 3.0)
    remainder_scale = tl.where(remainder == 0, 1.0,
                              tl.where(remainder == 1, 1.2599210498948732, 1.5874010519681994))
    scale = ((quotient + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    value = (estimate * remainder_scale) * scale
    value = tl.where(x < 0.0, -value, value)
    value = tl.where(magnitude == 0.0, x, value)
    tl.store(OUT + index, value, mask=active)


def validate_inputs(x, require_cuda=True):
    """Validate the complete public domain; also usable on CPU."""
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
    require(bool((torch.isfinite(x)).all().item()),
            "all input values must be finite")
    return {"valid": True, "size": x.numel()}


def run(x):
    """Return a fresh FP32 tensor approximating the real cube root of x."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _cbrt_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate the real cube root using the FP64 power library."""
    validate_inputs(x, require_cuda=False)
    value = x.to(torch.float64)
    return torch.sign(value) * torch.pow(torch.abs(value), 1.0 / 3.0)


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience examples only; all finite FP16 values are legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (128.0 * torch.rand((size,), generator=generator) - 64.0).to(torch.float16)
    return (x.to(device),)
