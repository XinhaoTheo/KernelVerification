import triton
import triton.language as tl
import torch

# Fused GELU kernel: single pass over input, computing
#   0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# entirely inside Triton. No separate activation step needed.

@triton.jit
def _gelu_kernel(
    x_ptr,       # pointer to input tensor
    out_ptr,     # pointer to output tensor
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):
    """
    Triton kernel implementing the GELU activation:
      GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Fused stages (single kernel pass):
      1. Load x from global memory
      2. Compute x^3
      3. Compute inner = sqrt(2/pi) * (x + 0.044715 * x^3)
      4. Compute tanh(inner) using tl.math.tanh
      5. Compute 0.5 * x * (1 + tanh(inner))
      6. Store result to global memory
    """
    # Block/program index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input; use fp32 for computation to maintain numerical accuracy
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU constants
    # sqrt(2/pi) ≈ 0.7978845608028654
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    COEFF: tl.constexpr = 0.044715

    # Compute x^3
    x2 = x * x
    x3 = x2 * x

    # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = SQRT_2_OVER_PI * (x + COEFF * x3)

    # tanh(inner)
    tanh_inner = tl.math.tanh(inner)

    # GELU(x) = 0.5 * x * (1 + tanh(inner))
    result = 0.5 * x * (1.0 + tanh_inner)

    # Cast back to the original dtype (bfloat16) before storing
    # We load the pointer element type from out_ptr
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the fused GELU Triton kernel.

    Fused computation (single kernel, single memory pass):
      - Load x
      - Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      - Store result

    Args:
        x: Input tensor of any shape, expected dtype bfloat16 on CUDA.

    Returns:
        Output tensor of same shape and dtype as input, with GELU applied.
    """
    # Validate input
    assert x.is_cuda, "Input tensor must be on CUDA device"

    # Allocate output tensor with same shape and dtype as input
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()

    # Choose block size (power of 2, good for memory coalescing)
    BLOCK_SIZE = 1024

    # Grid: one program per block of BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the fused GELU Triton kernel
    _gelu_kernel[grid](
        x,          # input pointer
        out,        # output pointer
        n_elements, # total elements
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out