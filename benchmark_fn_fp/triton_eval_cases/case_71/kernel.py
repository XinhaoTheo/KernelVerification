"""Row-wise first-maximum index via a packed integer-key reduction.

Original synthetic Triton implementation; not an upstream historical bug.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    value = tl.load(X + row * N + column, mask=column < N, other=0.0)
    raw = value.to(tl.uint16, bitcast=True).to(tl.uint32)
    raw = tl.where((raw & 0x7fff) == 0, 0, raw).to(tl.uint32)
    mask = tl.where((raw & 0x8000) != 0, 0xffff, 0x8000).to(tl.uint32)
    rank = raw ^ mask
    # Rank dominates; reversed column indices select the first equal value.
    tie = 0xffffffff - column.to(tl.uint32)
    key = (rank.to(tl.uint64) << 32) | tie.to(tl.uint64)
    key = tl.where(column < N, key, 0)
    best = tl.max(key, axis=0)
    winner = (0xffffffff - (best & 0xffffffff)).to(tl.int64)
    tl.store(OUT + row, winner)


def validate_inputs(x, require_cuda=True):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    require(type(x) is torch.Tensor, "ordinary torch.Tensor required")
    require(x.layout == torch.strided and x.is_contiguous(), "contiguous strided input required")
    require(not x.requires_grad and not x.is_inference(), "ordinary non-gradient input required")
    require(not x.is_neg() and not x.is_conj(), "logical view flags outside domain")
    require(not require_cuda or x.is_cuda, "CUDA input required")
    require(x.ndim == 2, "matrix [M,N] required")
    require(1 <= x.shape[0] <= 65536 and 1 <= x.shape[1] <= 4096
            and x.numel() <= 262144, "matrix dimensions outside domain")
    require(x.dtype == torch.float16, "FP16 input required")
    require(bool(torch.isfinite(x).all().item()), "all values must be finite")
    return {"valid": True, "shape": list(x.shape)}


def run(x):
    validate_inputs(x)
    m, n = x.shape
    with torch.cuda.device(x.device):
        output = torch.empty((m,), dtype=torch.int64, device=x.device)
        _argmax_kernel[(m,)](x, output, N=n, BLOCK=triton.next_power_of_2(n), num_warps=4)
    return output


def reference(x):
    """FP16-to-FP64 conversion is exact; numerical ties use the first index."""
    validate_inputs(x, require_cuda=False)
    return torch.argmax(x.to(torch.float64), dim=1)


def make_inputs(device="cuda", rows=17, columns=257, seed=0):
    """Convenience examples, not a restriction on the input domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, columns), generator=generator).to(torch.float16)
    return (x.to(device),)
