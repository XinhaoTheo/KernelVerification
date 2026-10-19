"""Elementwise softplus using stable tails and a range-reduced logarithm.

Original synthetic Triton implementation of a standard activation function.
This is not a reproduction of an upstream implementation or historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softplus_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    exponential = tl.exp(-tl.abs(x))
    s = 1.0 + exponential
    # Range-reduce log(s) around sqrt(2), then use its odd-power expansion.
    q = tl.div_rn(s - 1.4142135623730951, s + 1.4142135623730951)
    square = q * q
    polynomial = 0.2
    polynomial = 0.3333333333333333 + square * polynomial
    polynomial = 1.0 + square * polynomial
    correction = 0.34657359027997265 + (2.0 * q) * polynomial
    value = tl.maximum(x, 0.0) + correction
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
    """Return a fresh FP32 tensor approximating softplus of each stored input value."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _softplus_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate mathematical softplus with a stable FP64 expression."""
    validate_inputs(x, require_cuda=False)
    value = x.to(torch.float64)
    return torch.clamp(value, min=0.0) + torch.log1p(torch.exp(-torch.abs(value)))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (2.0 * torch.randn((size,), generator=generator)).to(torch.float16)
    return (x.to(device),)
