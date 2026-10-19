"""Elementwise complex division on finite binary16 components.

Original synthetic Triton implementation of standard complex arithmetic.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _divide(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    a = tl.load(X + 2 * index, mask=active, other=0).to(tl.float32)
    b = tl.load(X + 2 * index + 1, mask=active, other=0).to(tl.float32)
    c = tl.load(Y + 2 * index, mask=active, other=1).to(tl.float32)
    d = tl.load(Y + 2 * index + 1, mask=active, other=0).to(tl.float32)
    scale = c * c + d * d
    real_numerator = a * c + b * d
    imag_numerator = b * c - a * d
    real = tl.div_rn(real_numerator, scale)
    imag = tl.div_rn(imag_numerator, scale)
    tl.store(OUT + 2 * index, real, mask=active)
    tl.store(OUT + 2 * index + 1, imag, mask=active)


def validate_inputs(numerator, denominator, require_cuda=True):
    """Enforce the complete public tensor and component domain."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for value in (numerator, denominator):
        require(type(value) is torch.Tensor, "ordinary torch.Tensor required")
        require(value.layout == torch.strided and value.is_contiguous(), "contiguous strided input required")
        require(not value.requires_grad and not value.is_inference(), "ordinary non-gradient input required")
        require(not value.is_neg() and not value.is_conj(), "logical view flags are outside this domain")
        require(not require_cuda or value.is_cuda, "CUDA input required by run")
        require(value.dtype == torch.float16, "float16 input required")
        require(value.ndim == 2 and value.shape[1] == 2 and 1 <= value.shape[0] <= 131072, "shape must be [N,2], 1<=N<=131072")
        require(bool(torch.isfinite(value).all()), "all input components must be finite")
    require(numerator.shape == denominator.shape and numerator.device == denominator.device, "inputs must have the same shape and device")
    require(bool((denominator != 0).any(dim=1).all()), "each complex denominator must be nonzero")
    return {"valid": True, "size": numerator.shape[0]}


def run(numerator, denominator):
    """Return a fresh [N,2] FP32 tensor containing real and imaginary parts."""
    validate_inputs(numerator, denominator)
    n = numerator.shape[0]
    with torch.cuda.device(numerator.device):
        output = torch.empty((n, 2), dtype=torch.float32, device=numerator.device)
        _divide[(triton.cdiv(n, 256),)](numerator, denominator, output,
                                     N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(numerator, denominator):
    """Direct real formulas in FP64; supports CPU tensors for comparison."""
    validate_inputs(numerator, denominator, require_cuda=False)
    a, b = numerator.to(torch.float64).unbind(dim=1)
    c, d = denominator.to(torch.float64).unbind(dim=1)
    scale = c * c + d * d
    return torch.stack(((a * c + b * d) / scale, (b * c - a * d) / scale), dim=1)


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience examples, not restrictions on the legal component values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    numerator = torch.randn((size, 2), generator=generator).to(torch.float16)
    denominator = torch.randn((size, 2), generator=generator).to(torch.float16)
    zero = (denominator == 0).all(dim=1)
    denominator[zero, 0] = 1
    return numerator.to(device), denominator.to(device)
