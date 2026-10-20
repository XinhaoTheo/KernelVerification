"""Elementwise asinh using an odd polynomial and a logarithmic outer branch.

Original synthetic Triton implementation, not an upstream bug reproduction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _asinh_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    magnitude = tl.abs(x)
    local_magnitude = tl.minimum(magnitude, 1.05)
    square = local_magnitude * local_magnitude
    polynomial = -0.0011264356
    polynomial = polynomial * square + 0.0059712074
    polynomial = polynomial * square - 0.015329496
    polynomial = polynomial * square + 0.027586811
    polynomial = polynomial * square - 0.043983627
    polynomial = polynomial * square + 0.074917369
    polynomial = polynomial * square - 0.16666223
    polynomial = polynomial * square + 0.99999994
    local_value = local_magnitude * polynomial
    outer_value = tl.log(magnitude + tl.sqrt(1.0 + magnitude * magnitude))
    absolute_value = tl.where(magnitude <= 1.0, local_value, outer_value)
    value = tl.where(x < 0.0, -absolute_value, absolute_value)
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
    require(bool(torch.isfinite(x).all().item()),
            "all input values must be finite")
    return {"valid": True, "size": x.numel()}


def run(x):
    """Return a fresh FP32 tensor approximating asinh(x)."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _asinh_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate the mathematical target with the FP64 library asinh."""
    validate_inputs(x, require_cuda=False)
    return torch.asinh(x.to(torch.float64))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience examples only; all finite FP16 values are legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (8.0 * torch.rand((size,), generator=generator) - 4.0).to(torch.float16)
    return (x.to(device),)
