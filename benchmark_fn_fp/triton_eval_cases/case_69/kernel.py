"""Sequency-ordered integer Walsh-Hadamard transform.

Original synthetic Triton implementation of a standard signal transform.
This is not an upstream implementation or a claimed historical defect.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _transform_kernel(X, OUT, N: tl.constexpr, LOG_N: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, N)
    value = tl.load(X + row * N + lane).to(tl.int32)
    for stage in tl.static_range(0, LOG_N):
        step = 1 << stage
        peer = tl.gather(value, lane ^ step, axis=0)
        value = tl.where((lane & step) == 0, value + peer, peer - value)

    # Convert the coefficient index to the butterfly's natural row index.
    gray = lane ^ (lane >> 1)
    source_lane = tl.full((N,), 0, tl.int32)
    for bit in tl.static_range(0, LOG_N):
        source_lane = source_lane | (((gray >> bit) & 1) << (LOG_N - 1 - bit))

    result = tl.gather(value, source_lane, axis=0)
    tl.store(OUT + row * N + lane, result)


def validate_inputs(x, require_cuda=True):
    """Validate the complete public domain; also usable on CPU."""
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
    require(x.ndim == 2 and 1 <= x.shape[0] <= 128 and x.shape[1] in (32, 64, 128),
            "x must have shape [B,N], B in [1,128] and N in {32,64,128}")
    require(x.dtype == torch.int16, "int16 input required")
    return {"valid": True, "batch": x.shape[0], "width": x.shape[1]}


def run(x):
    """Return the unnormalized transform with coefficients in sequency order."""
    validate_inputs(x)
    batch, width = x.shape
    with torch.cuda.device(x.device):
        output = torch.empty((batch, width), dtype=torch.int32, device=x.device)
        _transform_kernel[(batch,)](
            x, output, N=width, LOG_N=width.bit_length() - 1, num_warps=4)
    return output


def reference(x):
    """Build the mathematical basis independently, then multiply in int64."""
    validate_inputs(x, require_cuda=False)
    width = x.shape[1]
    hadamard = torch.ones((1, 1), dtype=torch.int64, device=x.device)
    while hadamard.shape[0] < width:
        top = torch.cat((hadamard, hadamard), dim=1)
        bottom = torch.cat((hadamard, -hadamard), dim=1)
        hadamard = torch.cat((top, bottom), dim=0)
    changes = (hadamard[:, 1:] != hadamard[:, :-1]).sum(dim=1)
    sequency_rows = hadamard[torch.argsort(changes)]
    # int64 products and sums are exact throughout the complete int16 domain.
    result = (x.to(torch.int64)[:, None, :] * sequency_rows[None, :, :]).sum(dim=2)
    return result.to(torch.int32)


def make_inputs(device="cuda", batch=7, width=64, seed=0):
    """Examples only; all stored int16 values are legal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-32768, 32768, (batch, width), generator=generator, dtype=torch.int32)
    return (x.to(device=device, dtype=torch.int16),)
