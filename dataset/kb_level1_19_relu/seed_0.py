import triton
import triton.language as tl
import torch

# Fused operation: ReLU activation (single pass, elementwise)
# Stage 1: Load input element
# Stage 2: Apply ReLU (max(0, x)) in-kernel using Triton
# Stage 3: Store result
# No separate passes needed; fully fused in one kernel launch.

@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for ReLU activation.
    
    Computes out[i] = max(0, x[i]) for all elements.
    Each program instance handles BLOCK_SIZE contiguous elements.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply ReLU: max(0, x)
    zero = tl.zeros(x.shape, dtype=x.dtype)
    result = tl.maximum(zero, x)

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the fused ReLU Triton kernel.

    Fusion notes:
    - Single kernel pass: load -> ReLU (max(0, x)) -> store
    - No intermediate buffers; no separate activation pass needed.
    - Handles BF16 inputs natively via Triton's dtype inference.

    Args:
        x: Input tensor of any shape and dtype (BF16 supported).

    Returns:
        Output tensor of the same shape and dtype as input,
        with ReLU applied elementwise.
    """
    # Validate input
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()

    # Choose block size (power of 2, good for memory coalescing)
    BLOCK_SIZE = 1024

    # Grid: one program per block of BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel
    _relu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out