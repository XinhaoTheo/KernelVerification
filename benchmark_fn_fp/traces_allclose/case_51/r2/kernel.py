"""Fused fixed-point rescaling and dequantization for an integer GEMM epilogue.

Synthetic Triton implementation of the gemmlowp double-rounding arithmetic.
The arithmetic specification is inspired by gemmlowp fixedpoint/fixedpoint.h;
this is not a port of an upstream kernel or a claim of a historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rescale(ACC, MULT, SHIFT, ZERO, SCALE, OUT,
             ROWS: tl.constexpr, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < ROWS * CHANNELS
    channel = index % CHANNELS
    a = tl.load(ACC + index, mask=active, other=0)
    m = tl.load(MULT + channel, mask=active, other=1073741824)
    r = tl.load(SHIFT + channel, mask=active, other=0)
    z = tl.load(ZERO + channel, mask=active, other=0)
    scale = tl.load(SCALE + channel, mask=active, other=1.0)

    # Signed upper product and modulo-2^32 lower product.
    high = tl.inline_asm_elementwise(
        "mul.hi.s32 $0, $1, $2;", constraints="=r,r,r",
        args=[a, m], dtype=tl.int32, is_pure=True, pack=1)
    low = a.to(tl.uint32) * m.to(tl.uint32)
    low_a = low + tl.full((BLOCK,), 0x40000000, tl.uint32)
    carry_a = (low_a < low).to(tl.int32)

    negative = a < 0
    low_b = tl.where(
        r == 1, tl.where(negative, 0, 0x80000000),
        tl.where((r > 1) & negative, 0x80000000, 0)).to(tl.uint32)
    high_b = tl.where(
        r >= 2, (1 << tl.maximum(r - 2, 0)) - negative.to(tl.int32), 0)
    low_final = low_a + low_b
    carry_b = (low_final < low_a).to(tl.int32)
    high_final = high + high_b + carry_a + carry_b

    q0 = ((high_final.to(tl.uint32) << 1) | (low_final >> 31)).to(tl.int32)
    shifted = high_final >> tl.maximum(r - 1, 0)
    q = tl.where(r == 0, q0, shifted)
    q = tl.minimum(tl.maximum(q, -255), 255)
    q = tl.minimum(tl.maximum(q + z, -128), 127)
    result = (q - z).to(tl.float32) * scale
    tl.store(OUT + index, result, mask=active)


def validate_inputs(acc, multipliers, right_shifts, zero_points, output_scales,
                    require_cuda=True):
    """Validate the whole public domain; CPU validation is also available."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    buffers = (acc, multipliers, right_shifts, zero_points, output_scales)
    for tensor in buffers:
        require(type(tensor) is torch.Tensor, "ordinary torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(),
                "contiguous strided tensors required")
        require(not tensor.requires_grad and not tensor.is_inference(),
                "ordinary non-gradient buffers required")
        require(not tensor.is_neg() and not tensor.is_conj(),
                "logical view flags are outside this domain")
        require(not require_cuda or tensor.is_cuda, "CUDA tensors required by run")
    require(all(t.device == acc.device for t in buffers), "one device required")
    require(acc.ndim == 2 and acc.dtype == torch.int32, "acc must be int32 [M,C]")
    rows, channels = acc.shape
    require(1 <= rows <= 4096 and 1 <= channels <= 257,
            "M must be in [1,4096] and C in [1,257]")
    for tensor in (multipliers, right_shifts, zero_points):
        require(tensor.dtype == torch.int32 and tensor.shape == (channels,),
                "integer channel parameters must be int32 [C]")
    require(output_scales.dtype == torch.float32 and output_scales.shape == (channels,),
            "output_scales must be FP32 [C]")
    require(bool((multipliers >= (1 << 30)).all().item()),
            "Q31 multipliers must be in [2^30,2^31-1]")
    require(bool(((right_shifts >= 0) & (right_shifts <= 31)).all().item()),
            "right shifts must be in [0,31]")
    require(bool(((zero_points >= -128) & (zero_points <= 127)).all().item()),
            "zero points must be in [-128,127]")
    allowed_scale = torch.zeros_like(output_scales, dtype=torch.bool)
    for exponent in range(-8, 1):
        allowed_scale |= output_scales == 2.0 ** exponent
    require(bool(allowed_scale.all().item()), "scales must be powers of two from 2^-8 to 1")
    return {"valid": True, "rows": rows, "channels": channels}


def run(acc, multipliers, right_shifts, zero_points, output_scales):
    """Return a fresh FP32 dequantized epilogue result."""
    validate_inputs(acc, multipliers, right_shifts, zero_points, output_scales)
    rows, channels = acc.shape
    with torch.cuda.device(acc.device):
        out = torch.empty((rows, channels), dtype=torch.float32, device=acc.device)
        _rescale[(triton.cdiv(rows * channels, 128),)](
            acc, multipliers, right_shifts, zero_points, output_scales, out,
            ROWS=rows, CHANNELS=channels, BLOCK=128, num_warps=4)
    return out


def reference(acc, multipliers, right_shifts, zero_points, output_scales):
    """Independent scalar Python-integer definition of the two rounding stages."""
    validate_inputs(acc, multipliers, right_shifts, zero_points, output_scales,
                    require_cuda=False)
    matrix = acc.cpu().tolist()
    mult = multipliers.cpu().tolist()
    shifts = right_shifts.cpu().tolist()
    zeros = zero_points.cpu().tolist()
    scales = output_scales.cpu().tolist()
    result = []
    for row in matrix:
        out_row = []
        for a, m, r, z, scale in zip(row, mult, shifts, zeros, scales):
            product = a * m
            high_round = (product + (1 << 30)) // (1 << 31)
            magnitude = abs(high_round)
            if r:
                magnitude = (magnitude + (1 << (r - 1))) // (1 << r)
            scaled = -magnitude if high_round < 0 else magnitude
            quantized = min(127, max(-128, scaled + z))
            out_row.append((quantized - z) * scale)
        result.append(out_row)
    return torch.tensor(result, dtype=torch.float32, device=acc.device)


def make_inputs(device="cuda", rows=31, channels=33, seed=0):
    """Example generator only; the legal accumulator domain is all int32 values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    acc = torch.randint(-(1 << 31), 1 << 31, (rows, channels),
                        dtype=torch.int64, generator=generator).to(torch.int32)
    multipliers = torch.randint(1 << 30, 1 << 31, (channels,),
                                dtype=torch.int64, generator=generator).to(torch.int32)
    shifts = torch.randint(0, 32, (channels,), dtype=torch.int32, generator=generator)
    zeros = torch.randint(-128, 128, (channels,), dtype=torch.int32, generator=generator)
    exponents = torch.randint(-8, 1, (channels,), generator=generator)
    scales = torch.pow(torch.tensor(2.0), exponents).to(torch.float32)
    return tuple(t.to(device) for t in (acc, multipliers, shifts, zeros, scales))
