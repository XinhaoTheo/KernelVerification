import triton
import triton.language as tl
import torch

# Fused softmax kernel:
# Stage 1 (fused in one pass): find row max via online reduction
# Stage 2 (fused in second pass): compute exp(x - max) and accumulate sum
# Stage 3 (fused in third pass): normalize by dividing by sum
# All three stages are implemented in a single kernel function with three sequential loops over the row.

@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    num_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel.
    
    Each program handles one row of the input matrix.
    
    Fused stages:
    1. Pass 1: Compute row maximum (online reduction across BLOCK_SIZE chunks)
    2. Pass 2: Compute sum of exp(x - max) (online reduction across BLOCK_SIZE chunks)
    3. Pass 3: Write normalized output exp(x - max) / sum
    """
    row_idx = tl.program_id(axis=0)
    
    row_start = row_idx * stride_row
    
    # ----------------------------------------------------------------
    # Pass 1: Find the maximum value in this row (for numerical stability)
    # ----------------------------------------------------------------
    row_max = tl.full((), value=float('-inf'), dtype=tl.float32)
    
    col_offsets_base = tl.arange(0, BLOCK_SIZE)
    
    num_blocks = tl.cdiv(num_cols, BLOCK_SIZE)
    
    for block_id in tl.range(0, num_blocks):
        col_offsets = block_id * BLOCK_SIZE + col_offsets_base
        mask = col_offsets < num_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))
        x_vals_f32 = x_vals.to(tl.float32)
        block_max = tl.max(x_vals_f32, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # ----------------------------------------------------------------
    # Pass 2: Compute sum of exp(x - max)
    # ----------------------------------------------------------------
    exp_sum = tl.zeros((), dtype=tl.float32)
    
    for block_id in tl.range(0, num_blocks):
        col_offsets = block_id * BLOCK_SIZE + col_offsets_base
        mask = col_offsets < num_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))
        x_vals_f32 = x_vals.to(tl.float32)
        exp_vals = tl.exp(x_vals_f32 - row_max)
        # Zero out masked positions
        exp_vals = tl.where(mask, exp_vals, 0.0)
        exp_sum += tl.sum(exp_vals, axis=0)
    
    # ----------------------------------------------------------------
    # Pass 3: Normalize and store output
    # ----------------------------------------------------------------
    inv_sum = 1.0 / exp_sum
    
    for block_id in tl.range(0, num_blocks):
        col_offsets = block_id * BLOCK_SIZE + col_offsets_base
        mask = col_offsets < num_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))
        x_vals_f32 = x_vals.to(tl.float32)
        exp_vals = tl.exp(x_vals_f32 - row_max)
        out_vals = exp_vals * inv_sum
        # Cast back to input dtype (bfloat16)
        out_vals_bf16 = out_vals.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_start + col_offsets, out_vals_bf16, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax over dim=1 using a fused Triton kernel.
    
    Fused operations (all in one kernel, three sequential passes over the row):
      1. Row-wise maximum reduction (for numerical stability)
      2. exp(x - max) and sum reduction
      3. Division (normalization) and store
    
    Args:
        x: Input tensor of shape (batch_size, dim), bfloat16
        
    Returns:
        Softmax output tensor, same shape and dtype as input
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    
    batch_size, num_cols = x.shape
    
    # Allocate output
    out = torch.empty_like(x)
    
    # stride_row: number of elements between consecutive rows
    stride_row = x.stride(0)
    
    # BLOCK_SIZE: number of columns processed per Triton program iteration
    # Use 1024 to balance register pressure vs. loop overhead
    # For 393216 cols: 393216 / 1024 = 384 iterations per row
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (batch_size,)
    
    _softmax_kernel[grid](
        x,
        out,
        num_cols,
        stride_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out