"""Single-rounding per-channel integer requantization.

Original synthetic Triton implementation of a fixed-point GEMM epilogue.
The task specifies exact compatibility with the stated single-rounding rule.
This file is not an upstream kernel or a historical bug reproduction.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _rescale_scalar(a, m, r, z):
    product = a.to(tl.float64) * m.to(tl.float64)
    denominator = (tl.full((), 1, tl.int64) << (31 + r)).to(tl.float64)
    scaled = libdevice.div_rn(product, denominator)
    rounded = libdevice.floor(scaled + 0.5).to(tl.int64)
    bounded = tl.minimum(tl.maximum(rounded, -255), 255).to(tl.int32)
    return tl.minimum(tl.maximum(bounded + z, -128), 127).to(tl.int8)


@triton.jit
def _requantize(ACC, MULT, SHIFT, ZERO, OUT,
                ROWS: tl.constexpr, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < ROWS * CHANNELS
    channel = index % CHANNELS
    a = tl.load(ACC + index, mask=active, other=0)
    m = tl.load(MULT + channel, mask=active, other=1073741824)
    r = tl.load(SHIFT + channel, mask=active, other=0)
    z = tl.load(ZERO + channel, mask=active, other=0)
    value = _rescale_scalar(a, m, r, z)
    tl.store(OUT + index, value, mask=active)


def validate_inputs(acc, multipliers, right_shifts, zero_points, require_cuda=True):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    buffers = (acc, multipliers, right_shifts, zero_points)
    for tensor in buffers:
        require(type(tensor) is torch.Tensor, "ordinary torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(),
                "contiguous strided tensors required")
        require(not tensor.requires_grad and not tensor.is_inference(),
                "ordinary non-gradient buffers required")
        require(not tensor.is_neg() and not tensor.is_conj(), "logical view flags excluded")
        require(not require_cuda or tensor.is_cuda, "CUDA tensors required by run")
        require(tensor.dtype == torch.int32, "all inputs must be int32")
    require(all(t.device == acc.device for t in buffers), "one device required")
    require(acc.ndim == 2, "acc must have shape [M,C]")
    rows, channels = acc.shape
    require(1 <= rows <= 4096 and 1 <= channels <= 257,
            "M must be 1..4096 and C must be 1..257")
    for tensor in buffers[1:]:
        require(tensor.shape == (channels,), "channel parameters must have shape [C]")
    require(bool((multipliers >= (1 << 30)).all().item()), "multipliers must be 2^30..2^31-1")
    require(bool(((right_shifts >= 0) & (right_shifts <= 31)).all().item()),
            "right shifts must be 0..31")
    require(bool(((zero_points >= -128) & (zero_points <= 127)).all().item()),
            "zero points must be -128..127")
    return {"valid": True, "rows": rows, "channels": channels}


def run(acc, multipliers, right_shifts, zero_points):
    validate_inputs(acc, multipliers, right_shifts, zero_points)
    rows, channels = acc.shape
    with torch.cuda.device(acc.device):
        output = torch.empty((rows, channels), dtype=torch.int8, device=acc.device)
        _requantize[(triton.cdiv(rows * channels, 128),)](
            acc, multipliers, right_shifts, zero_points, output,
            ROWS=rows, CHANNELS=channels, BLOCK=128, num_warps=4,
            enable_fp_fusion=False)
    return output


def reference(acc, multipliers, right_shifts, zero_points):
    """Direct single-rounding definition, with exact signed int64 arithmetic."""
    validate_inputs(acc, multipliers, right_shifts, zero_points, require_cuda=False)
    a = acc.cpu().to(torch.int64)
    m = multipliers.cpu().to(torch.int64)
    r = right_shifts.cpu().to(torch.int64)
    z = zero_points.cpu().to(torch.int64)
    product = a * m
    bias = torch.bitwise_left_shift(torch.ones_like(r), 30 + r)
    rounded = torch.bitwise_right_shift(product + bias, 31 + r)
    return torch.clamp(rounded + z, -128, 127).to(torch.int8).to(acc.device)


def make_inputs(device="cuda", rows=37, channels=65, seed=0):
    """Example generator; every int32 accumulator belongs to the public domain."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(-(1 << 31), 1 << 31, (rows, channels),
                      generator=g, dtype=torch.int64).to(torch.int32)
    m = torch.randint(1 << 30, 1 << 31, (channels,), generator=g,
                      dtype=torch.int64).to(torch.int32)
    r = torch.randint(0, 32, (channels,), generator=g, dtype=torch.int32)
    z = torch.randint(-128, 128, (channels,), generator=g, dtype=torch.int32)
    return tuple(t.to(device) for t in (a, m, r, z))
