"""Unsigned index-to-coordinate conversion by reciprocal multiplication.

Original synthetic Triton adaptation of fast integer divmod techniques.
The output contains row/column coordinates for a width-seven layout.
"""
import torch
import triton
import triton.language as tl

_DIVISOR = 7
_SHIFT = 31 + (_DIVISOR - 1).bit_length()
_MULTIPLIER = ((1 << _SHIFT) + _DIVISOR - 1) // _DIVISOR


@triton.jit
def _unflatten(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr,
               MULTIPLIER: tl.constexpr, SHIFT: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    value = tl.load(X + index, mask=active, other=0).to(tl.uint64)
    quotient = (value * MULTIPLIER) >> SHIFT
    remainder = value - quotient * 7
    tl.store(OUT + 2 * index, quotient.to(tl.uint32), mask=active)
    tl.store(OUT + 2 * index + 1, remainder.to(tl.uint32), mask=active)


def validate_inputs(indices, require_cuda=True):
    """Enforce the complete public domain, including uint32 extrema."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(indices) is torch.Tensor, "ordinary torch.Tensor required")
    require(indices.layout == torch.strided and indices.is_contiguous(),
            "contiguous strided input required")
    require(not indices.requires_grad and not indices.is_inference(),
            "ordinary non-gradient input required")
    require(not indices.is_neg() and not indices.is_conj(),
            "logical view flags are outside this domain")
    require(not require_cuda or indices.is_cuda, "CUDA input required by run")
    require(indices.ndim == 1 and 1 <= indices.numel() <= 131072,
            "indices must have shape [N] with N in [1,131072]")
    require(indices.dtype == torch.uint32, "uint32 input required")
    return {"valid": True, "size": indices.numel()}


def run(indices):
    """Return unsigned row/column coordinates in a fresh [N,2] tensor."""
    validate_inputs(indices)
    n = indices.numel()
    with torch.cuda.device(indices.device):
        output = torch.empty((n, 2), dtype=torch.uint32, device=indices.device)
        _unflatten[(triton.cdiv(n, 256),)](
            indices, output, N=n, BLOCK=256, MULTIPLIER=_MULTIPLIER,
            SHIFT=_SHIFT, num_warps=4)
    return output


def reference(indices):
    """Exact direct integer division on an int64 widening of each uint32."""
    validate_inputs(indices, require_cuda=False)
    values = indices.to(torch.int64)
    rows = torch.div(values, 7, rounding_mode="floor")
    columns = torch.remainder(values, 7)
    return torch.stack((rows, columns), dim=1).to(torch.uint32)


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience examples; all uint32 values belong to the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(0, 1 << 32, (size,), dtype=torch.int64,
                           generator=generator).to(torch.uint32)
    return (values.to(device),)
