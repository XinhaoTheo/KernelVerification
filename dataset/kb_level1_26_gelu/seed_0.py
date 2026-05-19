import triton
import triton.language as tl
import torch

# GELU activation kernel
# Fused operation: single-pass elementwise GELU
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# This is the standard "tanh approximation" of GELU used widely in practice.
# All computation is done in a single Triton kernel pass — no separate ops needed.

@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton GELU kernel using the tanh approximation:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Cast to float32 for computation stability
    x_f32 = x.to(tl.float32)

    # GELU tanh approximation constants
    # sqrt(2/pi) ≈ 0.7978845608028654
    # coefficient for x^3 term: 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    # Compute inner term: sqrt(2/pi) * (x + 0.044715 * x^3)
    x_cubed = x_f32 * x_f32 * x_f32
    inner = sqrt_2_over_pi * (x_f32 + coeff * x_cubed)

    # Compute tanh(inner)
    # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    # Use tl.math.tanh for numerical stability
    tanh_val = tl.math.tanh(inner)

    # GELU(x) = 0.5 * x * (1 + tanh(inner))
    result_f32 = 0.5 * x_f32 * (1.0 + tanh_val)

    # Cast back to original dtype (bfloat16 in this case)
    result = result_f32.to(x.dtype)

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the GELU Triton kernel.

    Fused stages (single kernel pass):
      1. Load input element
      2. Cast to float32 for precision
      3. Compute tanh-approximation GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
      4. Cast back to input dtype (bfloat16)
      5. Store result

    Args:
        x: Input tensor of any shape, expected dtype bfloat16 on CUDA.

    Returns:
        Output tensor of same shape and dtype as input with GELU applied elementwise.
    """
    # Validate device
    assert x.is_cuda, "Input tensor must be on CUDA device"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()

    # Choose BLOCK_SIZE as a compile-time constant (power of 2)
    BLOCK_SIZE = 1024

    # 1D grid: one block per BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton GELU kernel
    _gelu_kernel[grid](
        x,          # input pointer
        out,        # output pointer
        n_elements, # total number of elements
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out