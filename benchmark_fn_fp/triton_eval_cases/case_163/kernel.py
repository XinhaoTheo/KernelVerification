"""Elementwise base-two exponential with integer range reduction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _exp2_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    integer = tl.floor(x + 0.5).to(tl.int32)
    fraction = x - integer.to(tl.float32)
    polynomial = 0.0013391
    polynomial = polynomial * fraction + 0.009676
    polynomial = polynomial * fraction + 0.055504
    polynomial = polynomial * fraction + 0.24022
    polynomial = polynomial * fraction + 0.69315
    polynomial = polynomial * fraction + 1.0
    scale = ((integer + 127) << 23).to(tl.float32, bitcast=True)
    value = scale * polynomial
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
    require(x.dtype == torch.bfloat16, "BF16 input required")
    require(bool(torch.isfinite(x).all().item()), "all input values must be finite")
    require(bool((x.abs() <= 10).all().item()), "all input values must be in [-10,10]")
    return {"valid": True, "size": x.numel()}


def run(x):
    """Return a fresh FP32 tensor approximating 2**x for every stored input."""
    validate_inputs(x)
    n = x.numel()
    with torch.cuda.device(x.device):
        output = torch.empty((n,), dtype=torch.float32, device=x.device)
        _exp2_kernel[(triton.cdiv(n, 256),)](
            x, output, N=n, BLOCK=256, num_warps=4, enable_fp_fusion=False)
    return output


def reference(x):
    """Evaluate mathematical exp2 independently using the FP64 library function."""
    validate_inputs(x, require_cuda=False)
    return torch.exp2(x.to(torch.float64))


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (20.0 * torch.rand((size,), generator=generator) - 10.0).to(torch.bfloat16)
    return (x.to(device),)
