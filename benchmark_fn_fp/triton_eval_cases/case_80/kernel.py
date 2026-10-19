"""CRC-32 with parallel four-byte chunks and polynomial composition.

Original synthetic Triton adaptation of standard reflected CRC arithmetic.
"""
import zlib
import torch
import triton
import triton.language as tl

_POLY = 0xEDB88320
_MASK = 0xFFFFFFFF


def _polynomial_product(left, right):
    result = 0
    for bit in range(31, -1, -1):
        if left & (1 << bit):
            result ^= right
        right = (right >> 1) ^ (_POLY if right & 1 else 0)
    return result


def _advance_factor(byte_count):
    result = 1
    power = 0x40000000
    exponent = byte_count * 8
    while exponent:
        if exponent & 1:
            result = _polynomial_product(power, result)
        power = _polynomial_product(power, power)
        exponent >>= 1
    return result


def _make_coefficients(width):
    factors = [_advance_factor(width - 4 * (chunk + 1)) for chunk in range(width // 4)]
    initial = _MASK
    for _ in range(width * 8):
        initial = (initial >> 1) ^ (_POLY if initial & 1 else 0)
    return factors + [initial ^ _MASK]


@triton.jit
def _polynomial_product_gpu(left, right):
    result = tl.full(left.shape, 0, tl.uint32)
    for bit in tl.static_range(31, -1, -1):
        result = result ^ tl.where((left >> bit) & 1, right, 0)
        right = (right >> 1) ^ tl.where(right & 1, 0xEDB88320, 0)
    return result


@triton.jit
def _xor(left, right):
    return left ^ right


@triton.jit
def _checksum_kernel(X, COEFFICIENTS, OUT, N: tl.constexpr, CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.arange(0, CHUNKS)
    state = tl.full((CHUNKS,), 0, tl.uint32)
    for byte in tl.static_range(0, 4):
        value = tl.load(X + row * N + chunk * 4 + byte).to(tl.uint32)
        state = state ^ value
        for bit in tl.static_range(0, 8):
            state = (state >> 1) ^ tl.where(state & 1, 0xEDB88320, 0)
    factor = tl.load(COEFFICIENTS + chunk).to(tl.uint32)
    contribution = _polynomial_product_gpu(factor, state)
    combined = tl.reduce(contribution, axis=0, combine_fn=_xor)
    correction = tl.load(COEFFICIENTS + CHUNKS).to(tl.uint32)
    tl.store(OUT + row, (combined ^ correction).to(tl.int64))


def validate_inputs(x, require_cuda=True):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(x) is torch.Tensor, "ordinary torch.Tensor required")
    require(x.layout == torch.strided and x.is_contiguous(), "contiguous strided input required")
    require(not x.requires_grad and not x.is_inference(), "ordinary non-gradient input required")
    require(not x.is_neg() and not x.is_conj(), "logical view flags are outside this domain")
    require(not require_cuda or x.is_cuda, "CUDA input required by run")
    require(x.ndim == 2 and 1 <= x.shape[0] <= 128 and x.shape[1] in (64, 128),
            "x must have shape [B,N], B in [1,128] and N in {64,128}")
    require(x.dtype == torch.uint8, "uint8 input required")
    return {"valid": True, "batch": x.shape[0], "width": x.shape[1]}


def run(x):
    validate_inputs(x)
    batch, width = x.shape
    with torch.cuda.device(x.device):
        coefficients = torch.tensor(_make_coefficients(width), dtype=torch.int64, device=x.device)
        output = torch.empty((batch,), dtype=torch.int64, device=x.device)
        _checksum_kernel[(batch,)](x, coefficients, output, N=width, CHUNKS=width // 4, num_warps=4)
    return output


def reference(x):
    """Independent sequential CRC from the Python standard library."""
    validate_inputs(x, require_cuda=False)
    rows = x.detach().cpu().tolist()
    values = [zlib.crc32(bytes(row)) & 0xFFFFFFFF for row in rows]
    return torch.tensor(values, dtype=torch.int64, device=x.device)


def make_inputs(device="cuda", batch=7, width=64, seed=0):
    """Examples only; every byte sequence in the public shape domain is legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, 256, (batch, width), generator=generator, dtype=torch.uint8)
    return (x.to(device),)
