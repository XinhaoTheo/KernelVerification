"""Forward cyclic NTT with fused butterfly stages.

Original synthetic Triton implementation, not an upstream cryptographic kernel.
"""
import torch
import triton
import triton.language as tl

_Q = 3329
_ROOT = 17
_R = 1 << 16
_R_INV = pow(_R, -1, _Q)


def _coefficient_product(left, right):
    return (left * right) % _Q


def _make_coefficients(width):
    """Pack the coefficient banks for each fused stage and optional last stage."""
    omega = pow(_ROOT, 256 // width, _Q)
    result = []
    half = 1
    for _ in range((width.bit_length() - 1) // 2):
        first = [(_R * pow(omega, j * width // (2 * half), _Q)) % _Q
                 for j in range(half)]
        second = [(_R * pow(omega, j * width // (4 * half), _Q)) % _Q
                  for j in range(half)]
        upper = [(_R * pow(omega, (j + half) * width // (4 * half), _Q)) % _Q
                 for j in range(half)]
        cross_low = [_coefficient_product(a, b) for a, b in zip(first, second)]
        cross_high = [_coefficient_product(a, b) for a, b in zip(first, upper)]
        result.extend(first + second + upper + cross_low + cross_high)
        half *= 4
    if (width.bit_length() - 1) % 2:
        result.extend([(_R * pow(omega, j, _Q)) % _Q for j in range(width // 2)])
    return result


@triton.jit
def _multiply(value, coefficient):
    product = value.to(tl.int64) * coefficient.to(tl.int64)
    correction = (product * 3327) & 65535
    reduced = (product + correction * 3329) >> 16
    return tl.where(reduced >= 3329, reduced - 3329, reduced).to(tl.int32)


@triton.jit
def _transform_kernel(X, COEFFICIENTS, OUT, N: tl.constexpr, LOG_N: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, N)
    source = tl.full((N,), 0, tl.int32)
    for bit in tl.static_range(0, LOG_N):
        source = source | (((lane >> bit) & 1) << (LOG_N - 1 - bit))
    value = tl.load(X + row * N + source).to(tl.int32)

    for stage in tl.static_range(0, LOG_N // 2):
        half = 1 << (2 * stage)
        local = lane % half
        base = (lane // (4 * half)) * (4 * half) + local
        quadrant = (lane // half) & 3
        a = tl.gather(value, base, axis=0)
        b = tl.gather(value, base + half, axis=0)
        c = tl.gather(value, base + 2 * half, axis=0)
        d = tl.gather(value, base + 3 * half, axis=0)
        start = 5 * (half - 1) // 3
        first = tl.load(COEFFICIENTS + start + local)
        second = tl.load(COEFFICIENTS + start + half + local)
        upper = tl.load(COEFFICIENTS + start + 2 * half + local)
        cross_low = tl.load(COEFFICIENTS + start + 3 * half + local)
        cross_high = tl.load(COEFFICIENTS + start + 4 * half + local)
        p = _multiply(b, first)
        c_low = _multiply(c, second)
        c_high = _multiply(c, upper)
        d_low = _multiply(d, cross_low)
        d_high = _multiply(d, cross_high)
        lower_sum = a + p + c_low + d_low
        lower_difference = a + p - c_low - d_low
        upper_sum = a - p + c_high - d_high
        upper_difference = a - p - c_high + d_high
        value = tl.where(quadrant == 0, lower_sum,
                         tl.where(quadrant == 1, upper_sum,
                                  tl.where(quadrant == 2, lower_difference, upper_difference)))
        value = (value + 2 * 3329) % 3329

    if LOG_N % 2:
        half = N // 2
        local = lane % half
        left = tl.gather(value, local, axis=0)
        right = tl.gather(value, local + half, axis=0)
        start = 5 * ((1 << (2 * (LOG_N // 2))) - 1) // 3
        coefficient = tl.load(COEFFICIENTS + start + local)
        weighted = _multiply(right, coefficient)
        value = tl.where(lane < half, left + weighted, left - weighted)
        value = (value + 3329) % 3329
    tl.store(OUT + row * N + lane, value)


def validate_inputs(x, require_cuda=True):
    """Validate the complete public field and tensor domain."""
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
    require(x.dtype == torch.int32, "int32 input required")
    require(bool(((x >= 0) & (x < 3329)).all()), "canonical field values 0..3328 required")
    return {"valid": True, "batch": x.shape[0], "width": x.shape[1]}


def run(x):
    validate_inputs(x)
    batch, width = x.shape
    with torch.cuda.device(x.device):
        coefficients = torch.tensor(_make_coefficients(width), dtype=torch.int32, device=x.device)
        output = torch.empty_like(x)
        _transform_kernel[(batch,)](
            x, coefficients, output, N=width, LOG_N=width.bit_length() - 1, num_warps=4)
    return output


def reference(x):
    """Direct modular DFT, independent of the coefficient banks and butterflies."""
    validate_inputs(x, require_cuda=False)
    width = x.shape[1]
    matrix = [[pow(17, (256 // width) * k * j, 3329) for j in range(width)]
              for k in range(width)]
    coefficients = torch.tensor(matrix, dtype=torch.int64, device=x.device)
    products = x.to(torch.int64)[:, None, :] * coefficients[None, :, :]
    return (products.sum(dim=2) % 3329).to(torch.int32)


def make_inputs(device="cuda", batch=7, width=64, seed=0):
    """Examples only; every canonical field-valued row is legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, 3329, (batch, width), generator=generator, dtype=torch.int32)
    return (x.to(device),)
