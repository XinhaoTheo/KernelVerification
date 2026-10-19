"""Elementwise GELU using a Gaussian-tail approximation.

Original Triton adaptation of the Abramowitz--Stegun five-coefficient formula
for the Gaussian tail in GELU. This is a synthetic benchmark implementation,
not a claimed historical bug.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    magnitude = tl.abs(x)
    normalized = magnitude * 0.7071067811865475
    t = tl.div_rn(1.0, 1.0 + 0.3276 * normalized)
    polynomial = 1.06141 * t - 1.45315
    polynomial = polynomial * t + 1.42141
    polynomial = polynomial * t - 0.28450
    polynomial = polynomial * t + 0.25483
    polynomial = polynomial * t
    envelope = tl.exp(-(0.5 * (magnitude * magnitude)))
    tail = 0.5 * (polynomial * envelope)
    probability = tl.where(x < 0.0, tail, 1.0 - tail)
    value = x * probability
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
    """Return a fresh FP32 tensor approximating GELU of each stored input value."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _gelu_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate the mathematical GELU definition using the FP64 erfc function."""
    validate_inputs(x, require_cuda=False)
    value = x.to(torch.float64)
    return (0.5 * value) * torch.erfc(-value / math.sqrt(2.0))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (2.0 * torch.randn((size,), generator=generator)).to(torch.float16)
    return (x.to(device),)
