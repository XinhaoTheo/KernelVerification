"""Temporally fused, separable finite-domain binomial diffusion.

Synthetic benchmark implementation of a standard image/stencil operation.
This is not an upstream project's kernel or an asserted historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _extend_coordinate(position, SIZE: tl.constexpr):
    coordinate = position
    sign = 1.0
    valid = (position >= 0) & (position < SIZE)
    return coordinate, sign, valid


@triton.jit
def _sweep_four_steps(X, Y, H: tl.constexpr, W: tl.constexpr,
                      VERTICAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // W
    columns = offsets % W
    active = offsets < H * W
    coefficients: tl.constexpr = (1, 8, 28, 56, 70, 56, 28, 8, 1)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for tap in tl.static_range(9):
        if VERTICAL:
            source, sign, inside = _extend_coordinate(rows + tap - 4, H)
            address = source * W + columns
        else:
            source, sign, inside = _extend_coordinate(columns + tap - 4, W)
            address = rows * W + source
        value = tl.load(X + address, mask=active & inside, other=0.0).to(tl.float32)
        accumulator += value * sign * (coefficients[tap] / 256.0)
    tl.store(Y + offsets, accumulator, mask=active)


def validate_inputs(x, steps, require_cuda=True):
    """Check the complete public input domain, also usable on CPU."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(x) is torch.Tensor, "ordinary torch.Tensor required")
    require(x.layout == torch.strided and x.is_contiguous(), "contiguous strided input required")
    require(not x.requires_grad and not x.is_inference(), "ordinary non-gradient buffer required")
    require(not x.is_neg() and not x.is_conj(), "logical view flags are outside this domain")
    require(not require_cuda or x.is_cuda, "CUDA input required by run")
    require(x.ndim == 2, "expected x[H,W]")
    h, w = x.shape
    require(1 <= h <= 129 and 1 <= w <= 129, "H and W must be in [1,129]")
    require(x.dtype == torch.float16, "x must be FP16")
    require(bool(torch.isfinite(x).all().item()), "x must be finite")
    require(bool((x.abs() <= 1).all().item()), "input magnitude must be at most one")
    require(type(steps) is int and steps in (4, 8), "steps must be the Python integer 4 or 8")
    return {"valid": True, "height": h, "width": w, "steps": steps}


def run(x, steps):
    """Return a fresh FP32 field after the requested diffusion steps."""
    validate_inputs(x, steps)
    h, w = x.shape
    with torch.cuda.device(x.device):
        current = x
        for _ in range(steps // 4):
            horizontal = torch.empty((h, w), dtype=torch.float32, device=x.device)
            output = torch.empty((h, w), dtype=torch.float32, device=x.device)
            grid = (triton.cdiv(h * w, 128),)
            _sweep_four_steps[grid](current, horizontal, H=h, W=w,
                                    VERTICAL=False, BLOCK=128, num_warps=4)
            _sweep_four_steps[grid](horizontal, output, H=h, W=w,
                                    VERTICAL=True, BLOCK=128, num_warps=4)
            current = output
    return current


def reference(x, steps):
    """Direct FP64 three-by-three update, with a new zero exterior each step."""
    validate_inputs(x, steps, require_cuda=False)
    h, w = x.shape
    current = x.double()
    coefficients = (1.0, 2.0, 1.0)
    for _ in range(steps):
        padded = torch.zeros((h + 2, w + 2), dtype=torch.float64, device=x.device)
        padded[1:h + 1, 1:w + 1] = current
        output = torch.zeros_like(current)
        for row in range(3):
            for column in range(3):
                output += (coefficients[row] * coefficients[column] / 16.0) * padded[row:row + h, column:column + w]
        current = output
    return current


def make_inputs(device="cuda", height=31, width=33, steps=4, seed=0):
    """Convenience example generator, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (2.0 * torch.rand((height, width), generator=generator) - 1.0).half()
    return x.to(device), steps
