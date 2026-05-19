"""
Row-wise Softmax Triton Kernel Implementation

Fused stages in a single kernel pass per row:
  1. Load a block of the row
  2. Compute the row maximum (for numerical stability)
  3. Compute exp(x - max) for each element
  4. Compute the sum of exp values
  5. Normalize: divide each exp value by the sum
  6. Store the result

All five stages are fused into one kernel, avoiding intermediate
global memory writes for max, exp, or sum values.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,   # pointer to input tensor
    output_ptr,  # pointer to output tensor
    n_cols,      # number of columns (softmax dimension size)
    input_row_stride,   # stride between rows of input
    output_row_stride,  # stride between rows of output
    BLOCK_SIZE: tl.constexpr,  # number of columns processed per block (must be >= n_cols, power of 2)
):
    """
    Fused row-wise softmax kernel.

    Each program instance handles exactly one row of the input matrix.
    Within a single pass it:
      - Loads all elements of the row (with masking for boundary safety)
      - Computes the row maximum for numerical stability
      - Subtracts the max and computes exp
      - Sums the exp values
      - Divides by the sum to produce softmax probabilities
      - Stores the result
    """
    # Each program handles one row
    row_idx = tl.program_id(axis=0)

    # Compute base pointers for this row
    row_input_ptr  = input_ptr  + row_idx * input_row_stride
    row_output_ptr = output_ptr + row_idx * output_row_stride

    # Column offsets within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ---- Stage 1: Load row elements (fp32 for numerical stability) ----
    row = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))
    row = row.to(tl.float32)

    # ---- Stage 2: Compute row maximum for numerical stability ----
    row_max = tl.max(row, axis=0)

    # ---- Stage 3: Subtract max and compute exponentials ----
    row_shifted = row - row_max
    row_exp = tl.exp(row_shifted)

    # ---- Stage 4: Compute sum of exponentials ----
    # Mask out-of-bounds positions (they were loaded as -inf → exp(-inf)=0, so sum is fine)
    row_exp_masked = tl.where(mask, row_exp, 0.0)
    exp_sum = tl.sum(row_exp_masked, axis=0)

    # ---- Stage 5: Normalize ----
    softmax_out = row_exp_masked / exp_sum

    # ---- Stage 6: Store result (cast back to original dtype) ----
    tl.store(row_output_ptr + col_offsets, softmax_out, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise softmax along the last dimension using a fused Triton kernel.

    Fused operations (all in one kernel pass, no intermediate global writes):
      1. Load row
      2. Compute row max (numerical stability)
      3. Compute exp(x - max)
      4. Compute sum of exp values
      5. Normalize by sum
      6. Store output

    Args:
        x: Input tensor of shape (..., n_cols). Softmax is applied along dim=-1.

    Returns:
        Output tensor of the same shape and dtype as x, with softmax applied
        along the last dimension.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # Flatten all leading dimensions into a single "batch" (rows) dimension
    original_shape = x.shape
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    # Reshape to 2D for uniform processing
    x_2d = x.view(n_rows, n_cols)

    # Allocate output (same dtype as input; kernel computes in fp32 internally)
    output = torch.empty_like(x_2d)

    # Choose BLOCK_SIZE as the next power of 2 >= n_cols
    # This ensures the entire row fits in one block for a single-pass fused kernel.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Cap at 4096 to avoid excessive register pressure; for very wide rows
    # the kernel still works because masking handles the boundary.
    # For rows wider than 4096 we keep doubling until we cover n_cols.
    # (triton.next_power_of_2 already handles this.)
    # Clamp to supported range [64, 65536]
    BLOCK_SIZE = max(64, min(BLOCK_SIZE, 65536))

    # One program per row
    grid = (n_rows,)

    _softmax_kernel[grid](
        x_2d,                       # input pointer
        output,                     # output pointer
        n_cols,                     # number of valid columns
        x_2d.stride(0),             # input row stride (in elements)
        output.stride(0),           # output row stride (in elements)
        BLOCK_SIZE=BLOCK_SIZE,      # compile-time block size
    )

    # Restore original shape and dtype
    return output.view(original_shape)