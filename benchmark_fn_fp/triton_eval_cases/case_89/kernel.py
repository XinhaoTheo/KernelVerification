"""Synthetic packed real FFT using a radix-2 complex transform."""
import torch
import triton
import triton.language as tl

# Cosine table for the first quadrant of a 256-point transform.
_COSINE = (
    1.0, 0.9996988186962042, 0.9987954562051724, 0.9972904566786902,
    0.9951847266721969, 0.99247953459871, 0.989176509964781, 0.9852776423889412,
    0.9807852804032304, 0.9757021300385286, 0.970031253194544, 0.9637760657954398,
    0.9569403357322088, 0.9495281805930367, 0.9415440651830208, 0.9329927988347388,
    0.9238795325112867, 0.9142097557035307, 0.9039892931234433, 0.8932243011955153,
    0.881921264348355, 0.8700869911087115, 0.8577286100002721, 0.8448535652497071,
    0.8314696123025452, 0.8175848131515837, 0.8032075314806449, 0.7883464276266062,
    0.773010453362737, 0.7572088465064846, 0.7409511253549591, 0.7242470829514669,
    0.7071067811865476, 0.6895405447370669, 0.6715589548470184, 0.6531728429537768,
    0.6343932841636455, 0.6152315905806268, 0.5956993044924334, 0.5758081914178453,
    0.5555702330196022, 0.5349976198870973, 0.5141027441932218, 0.49289819222978404,
    0.47139673682599764, 0.4496113296546066, 0.4275550934302821, 0.40524131400498986,
    0.3826834323650898, 0.35989503653498817, 0.33688985339222005, 0.31368174039889146,
    0.2902846772544624, 0.26671275747489837, 0.2429801799032639, 0.2191012401568698,
    0.19509032201612828, 0.17096188876030122, 0.14673047445536175, 0.1224106751992162,
    0.0980171403295606, 0.07356456359966743, 0.049067674327418015, 0.024541228522912288,
    0.0,
)


@triton.jit
def _reverse(index, LOG_M: tl.constexpr):
    reversed_index = tl.full(index.shape, 0, tl.int32)
    for bit in tl.static_range(LOG_M):
        reversed_index = (reversed_index << 1) | ((index >> bit) & 1)
    return reversed_index


@triton.jit
def _twiddle(TABLE, phase):
    folded = tl.minimum(phase, 128 - phase)
    cosine = tl.load(TABLE + folded)
    cosine = tl.where(phase <= 64, cosine, -cosine)
    sine = tl.load(TABLE + tl.abs(64 - phase))
    return cosine, -sine


@triton.jit
def _realfft(X, TABLE, OUT, N: tl.constexpr, M: tl.constexpr, LOG_M: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, M)
    real = tl.load(X + row * N + 2 * k).to(tl.float64)
    imag = tl.load(X + row * N + 2 * k + 1).to(tl.float64)
    # Decimation in frequency leaves the complex transform in bit-reversed order.
    for stage in tl.static_range(LOG_M):
        width = M >> stage
        half = width >> 1
        left = k & ~half
        right = k | half
        ar = tl.gather(real, left, axis=0)
        ai = tl.gather(imag, left, axis=0)
        br = tl.gather(real, right, axis=0)
        bi = tl.gather(imag, right, axis=0)
        wr, wi = _twiddle(TABLE, (k & (half - 1)) * (256 // width))
        dr = ar - br
        di = ai - bi
        lower = (k & half) == 0
        real = tl.where(lower, ar + br, dr * wr - di * wi)
        imag = tl.where(lower, ai + bi, dr * wi + di * wr)
    address = _reverse(k, LOG_M)
    partner = _reverse((-k) & (M - 1), LOG_M)
    ar = tl.gather(real, address, axis=0)
    ai = tl.gather(imag, address, axis=0)
    br = tl.gather(real, partner, axis=0)
    bi = tl.gather(imag, partner, axis=0)
    # Reconstruct the real transform from the packed complex transform.
    wr, wi = _twiddle(TABLE, k * (256 // N))
    sr = ar + br
    si = ai - bi
    dr = ar - br
    di = ai + bi
    xr = 0.5 * (sr + (wr * di + wi * dr))
    xi = 0.5 * (si - (wr * dr - wi * di))
    tl.store(OUT + (row * (M + 1) + k) * 2, xr.to(tl.float32))
    tl.store(OUT + (row * (M + 1) + k) * 2 + 1, xi.to(tl.float32))
    tl.store(OUT + (row * (M + 1) + M + 0 * k) * 2, (ar - ai).to(tl.float32), mask=k == 0)
    tl.store(OUT + (row * (M + 1) + M + 0 * k) * 2 + 1, 0.0, mask=k == 0)


def validate_inputs(values, require_cuda=True):
    """Validate the public domain. Nonzero storage offsets are permitted."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(values) is torch.Tensor, "ordinary torch.Tensor required")
    require(values.layout == torch.strided and values.is_contiguous(), "contiguous strided input required")
    require(not values.requires_grad and not values.is_inference(), "ordinary non-gradient input required")
    require(not values.is_neg() and not values.is_conj(), "logical view flags are outside this domain")
    require(not require_cuda or values.is_cuda, "CUDA input required by run")
    require(values.dtype == torch.float16, "float16 input required")
    require(values.ndim == 2 and 1 <= values.shape[0] <= 256 and values.shape[1] in (64, 128, 256), "shape must be [B,N], 1<=B<=256, N in {64,128,256}")
    require(bool(torch.isfinite(values).all()), "all input values must be finite")
    return {"valid": True, "batch": values.shape[0], "width": values.shape[1]}


def run(values):
    """Return real and imaginary components in a fresh FP32 tensor."""
    validate_inputs(values)
    batch, n = values.shape
    m = n // 2
    with torch.cuda.device(values.device):
        table = torch.tensor(_COSINE, dtype=torch.float64, device=values.device)
        output = torch.empty((batch, m + 1, 2), dtype=torch.float32, device=values.device)
        _realfft[(batch,)](values, table, output, N=n, M=m, LOG_M=m.bit_length() - 1,
                          num_warps=4, enable_fp_fusion=False)
    return output


def reference(values):
    """A separate library transform in complex128, returned as FP64 components."""
    validate_inputs(values, require_cuda=False)
    result = torch.fft.rfft(values.to(torch.float64), dim=1, norm="backward")
    return torch.stack((result.real, result.imag), dim=-1)


def make_inputs(device="cuda", batch=7, width=128, seed=0):
    """Examples only; the contract includes every finite FP16 input."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((batch, width), generator=generator).to(torch.float16)
    return (values.to(device),)
