"""Elementwise sine using quadrant reduction and polynomial evaluation.

Original Triton adaptation of split-constant trigonometric range reduction.
This is a synthetic benchmark implementation, not a claimed historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sine_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    q = tl.floor(x * 0.63661977236758134308 + 0.5)
    r_hi = x - q * 1.5703125
    r = r_hi - q * 0.00048382679489661923
    z = r * r
    sin_poly = z * (1.0 / 362880.0) - 1.0 / 5040.0
    sin_poly = z * sin_poly + 1.0 / 120.0
    sin_poly = z * sin_poly - 1.0 / 6.0
    sine = r + (r * z) * sin_poly
    cos_poly = z * (1.0 / 40320.0) - 1.0 / 720.0
    cos_poly = z * cos_poly + 1.0 / 24.0
    cos_poly = z * cos_poly - 0.5
    cosine = 1.0 + z * cos_poly
    quadrant = q.to(tl.int32) & 3
    value = tl.where((quadrant & 1) == 0, sine, cosine)
    value = tl.where((quadrant & 2) == 0, value, -value)
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
    """Return a fresh FP32 tensor approximating sine of each stored input value."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _sine_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate sine in FP64 on the exact stored FP16 input values."""
    validate_inputs(x, require_cuda=False)
    return torch.sin(x.to(torch.float64))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (32.0 * torch.randn((size,), generator=generator)).to(torch.float16)
    return (x.to(device),)
