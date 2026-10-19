"""Elementwise FP16 sigmoid with explicit internal arithmetic precision.

Original synthetic Triton adaptation of the standard logistic activation.
This file is not presented as an upstream historical bug or patch.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _sigmoid(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < N
    x = tl.load(X + index, mask=active, other=0.0).to(tl.float32)
    exponential = libdevice.exp(-tl.abs(x))
    denominator = 1.0 + exponential
    numerator = tl.where(x < 0.0, exponential, 1.0)
    value = libdevice.div_rn(numerator, denominator)
    rounded = tl.inline_asm_elementwise(
        "cvt.rn.f16.f32 $0, $1;", constraints="=h,f",
        args=[value], dtype=tl.float16, is_pure=True, pack=1)
    tl.store(OUT + index, rounded, mask=active)


def validate_inputs(x, require_cuda=True):
    """Validate the complete public domain; CPU inputs are allowed for reference."""
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
    require(x.dtype == torch.float16, "FP16 input required")
    require(bool(torch.isfinite(x).all().item()), "all input values must be finite")
    return {"valid": True, "size": x.numel()}


def run(x):
    """Return sigmoid values in a fresh FP16 tensor."""
    validate_inputs(x)
    with torch.cuda.device(x.device):
        output = torch.empty_like(x)
        _sigmoid[(triton.cdiv(x.numel(), 256),)](
            x, output, N=x.numel(), BLOCK=256, num_warps=4,
            enable_fp_fusion=False)
    return output


def reference(x):
    """CPU FP64 sigmoid, then direct binary16 round-to-nearest-even packing."""
    import struct
    validate_inputs(x, require_cuda=False)
    values = x.detach().to(device="cpu", dtype=torch.float64)
    payload = bytearray().join(struct.pack("<e", value)
                               for value in torch.sigmoid(values).tolist())
    result = torch.frombuffer(payload, dtype=torch.float16).clone()
    return result.to(x.device)


def make_inputs(device="cuda", size=4093, seed=0):
    """Convenience inputs; all finite FP16 values belong to the legal domain."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (3.0 * torch.randn((size,), generator=generator)).to(torch.float16)
    return (x.to(device),)
