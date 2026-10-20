"""Elementwise log1p with exponent reduction and a short odd polynomial.

Original synthetic Triton implementation, not an upstream bug reproduction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _log1p_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    u = 1.0 + x
    correction = tl.div_rn(x - (u - 1.0), u)
    encoding = u.to(tl.uint32, bitcast=True)
    exponent = ((encoding >> 23) & 255).to(tl.int32) - 127
    mantissa = ((encoding & 0x007fffff) | 0x3f800000).to(tl.float32, bitcast=True)
    upper = mantissa > 1.65
    mantissa = tl.where(upper, mantissa * 0.5, mantissa)
    exponent = exponent + upper.to(tl.int32)
    z = tl.div_rn(mantissa - 1.0, mantissa + 1.0)
    square = z * z
    polynomial = 0.142857
    polynomial = polynomial * square + 0.2
    polynomial = polynomial * square + 0.33333
    polynomial = 1.0 + square * polynomial
    local_log = (2.0 * z) * polynomial
    high = exponent.to(tl.float32) * 0.69313812256
    low = exponent.to(tl.float32) * 0.0000090580006145
    value = high + (local_log + (low + correction))
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
    require(bool((torch.isfinite(x) & (x > -1.0)).all().item()),
            "all input values must be finite and greater than -1")
    return {"valid": True, "size": x.numel()}


def run(x):
    """Return a fresh FP32 tensor approximating log(1+x)."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _log1p_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate the mathematical target with the FP64 library log1p."""
    validate_inputs(x, require_cuda=False)
    return torch.log1p(x.to(torch.float64))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience examples only; all finite FP16 values greater than -1 are legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (4.0 * torch.rand((size,), generator=generator) - 0.75).to(torch.float16)
    return (x.to(device),)
