"""Elementwise bounded normalization using refined reciprocal square root.

Original synthetic Triton implementation of x/sqrt(1+x*x), the first
component of a normalized vector (x, 1). Not an upstream historical defect.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    squared_length = 1.0 + x * x
    # Approximate the reciprocal square root from its exponent and mantissa.
    encoding = squared_length.to(tl.int32, bitcast=True)
    initial = 0x5F3759DF - (encoding >> 1)
    inverse_length = initial.to(tl.float32, bitcast=True)
    half_squared_length = 0.5 * squared_length
    for _ in tl.static_range(3):
        inverse_length = inverse_length * (
            1.5 - half_squared_length * (inverse_length * inverse_length))
    value = x * inverse_length
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
    """Return the FP32 first coordinate of each normalized vector (x, 1)."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _normalize_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate the mathematical normalization in FP64."""
    validate_inputs(x, require_cuda=False)
    value = x.to(torch.float64)
    return value / torch.sqrt(1.0 + value * value)


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (8.0 * torch.randn((size,), generator=generator)).to(torch.float16)
    return (x.to(device),)
