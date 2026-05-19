import triton
import triton.language as tl
import torch

# Fused operation: sum reduction over a specified dimension with keepdim=True
# The kernel handles the reduction in a single pass:
#   1. Each program handles one (batch, col) pair, reducing over the specified dimension
#   2. Data is loaded in blocks along the reduction dimension
#   3. Partial sums are accumulated in fp32 for numerical stability
#   4. Final result is stored in bf16

@triton.jit
def _sum_reduce_dim1_kernel(
    x_ptr,          # Input tensor pointer
    out_ptr,        # Output tensor pointer
    B,              # batch size (dim 0)
    R,              # reduction dimension size (dim 1)
    C,              # column dimension size (dim 2)
    stride_b,       # stride for batch dimension
    stride_r,       # stride for reduction dimension
    stride_c,       # stride for column dimension
    out_stride_b,   # output stride for batch dimension
    out_stride_c,   # output stride for column dimension
    BLOCK_R: tl.constexpr,  # Block size along reduction dimension
    BLOCK_C: tl.constexpr,  # Block size along column dimension
):
    """
    Triton kernel for sum reduction over dim=1 with keepdim=True.
    
    Grid: (cdiv(C, BLOCK_C), B) - each program handles BLOCK_C columns for one batch.
    
    Fused stages:
      - Load input blocks along reduction dimension
      - Accumulate sum in fp32 for numerical stability
      - Store result (cast back to input dtype)
    """
    # Program IDs
    pid_c = tl.program_id(axis=0)  # Column block index
    pid_b = tl.program_id(axis=1)  # Batch index

    # Column offsets for this program
    col_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = col_offsets < C

    # Initialize accumulator in fp32 for numerical stability
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Base pointer for this batch
    batch_base = pid_b * stride_b

    # Iterate over the reduction dimension in blocks
    for r_start in tl.range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offsets < R

        # 2D mask: [BLOCK_R, BLOCK_C]
        mask_2d = r_mask[:, None] & col_mask[None, :]

        # Compute flat indices: batch_base + r * stride_r + c * stride_c
        # Shape: [BLOCK_R, BLOCK_C]
        indices = batch_base + r_offsets[:, None] * stride_r + col_offsets[None, :] * stride_c

        # Load values
        vals = tl.load(x_ptr + indices, mask=mask_2d, other=0.0)

        # Accumulate sum along reduction dimension (axis=0 of the 2D block)
        acc = acc + tl.sum(vals.to(tl.float32), axis=0)

    # Store results
    # Output shape: [B, 1, C], so output index = pid_b * out_stride_b + 0 * out_stride_r + col * out_stride_c
    out_indices = pid_b * out_stride_b + col_offsets
    tl.store(out_ptr + out_indices, acc.to(x_ptr.dtype.element_ty), mask=col_mask)


def kernel_function(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs sum reduction over the specified dimension with keepdim=True.
    
    This is a single-pass fused Triton kernel that:
      1. Iterates over the reduction dimension in blocks
      2. Accumulates partial sums in fp32 for numerical stability
      3. Stores the result cast back to the input dtype
    
    Args:
        x: Input tensor of shape (B, R, C) with dtype bfloat16
        dim: Dimension to reduce over (currently optimized for dim=1)
    
    Returns:
        Output tensor with keepdim=True, shape (B, 1, C) for dim=1
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    assert dim == 1, f"This kernel is optimized for dim=1 reduction, got dim={dim}"
    
    B, R, C = x.shape
    
    # Allocate output tensor with keepdim=True
    out_shape = list(x.shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Output is contiguous with shape [B, 1, C]
    # out_stride_b = C, out_stride_c = 1 (since middle dim is 1)
    out_stride_b = C  # stride over batch in output (1 * C = C)
    out_stride_c = 1  # stride over columns in output
    
    # Block sizes - tuned for the problem dimensions
    # BLOCK_C: number of columns processed per program
    # BLOCK_R: number of reduction steps per inner loop iteration
    BLOCK_C = 128
    BLOCK_R = 64
    
    # Grid: one program per (batch, column_block)
    grid = (triton.cdiv(C, BLOCK_C), B)
    
    _sum_reduce_dim1_kernel[grid](
        x,                      # x_ptr
        out,                    # out_ptr
        B,                      # B
        R,                      # R (reduction dim size)
        C,                      # C
        x.stride(0),            # stride_b
        x.stride(1),            # stride_r
        x.stride(2),            # stride_c
        out_stride_b,           # out_stride_b
        out_stride_c,           # out_stride_c
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
    )
    
    return out