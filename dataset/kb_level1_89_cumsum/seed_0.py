import triton
import triton.language as tl
import torch

# Fused note: The prefix scan is split into 3 passes (local scan, carry scan, carry add)
# because a single-pass scan across 32768 elements requires inter-block communication
# that cannot be done in one kernel launch without atomic-based approaches.
# The three passes are kept minimal and tight.

@triton.jit
def _cumsum_phase1_kernel(
    x_ptr,       # input pointer
    out_ptr,     # output pointer (local prefix sums)
    carry_ptr,   # workspace: total sum of each block
    N,           # number of elements per row
    stride_row,  # stride between rows
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: For each (row, block_col) tile, compute local prefix sum
    and store the block's total sum into carry_ptr.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    num_col_blocks = tl.num_programs(1)

    block_start = pid_col * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    row_offset = pid_row * stride_row
    x_vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute local prefix sum using sequential scan
    # Triton doesn't have a native prefix scan, so we do it manually
    # using a log-step parallel scan pattern
    v = x_vals

    # Parallel prefix sum (Kogge-Stone style) within the block
    offset1 = tl.arange(0, BLOCK_SIZE)

    # Step 1: stride 1
    v1 = tl.where(offset1 >= 1, tl.shift(v, 1, 0), tl.zeros_like(v))  # won't work directly
    # Use sequential approach instead for correctness
    # We'll do a simple iterative approach
    # Actually, let's use the standard Triton approach with associative scan

    # Re-implement with explicit loop unrolling via log2 steps
    # Kogge-Stone parallel prefix sum
    stride = 1
    for _ in tl.static_range(0, 15):  # log2(32768) = 15, but BLOCK_SIZE can be smaller
        shifted = tl.shift(v, stride, 0)
        idx = tl.arange(0, BLOCK_SIZE)
        v = tl.where(idx >= stride, v + shifted, v)
        stride = stride * 2

    # Store local prefix sums
    tl.store(out_ptr + row_offset + offsets, v.to(tl.bfloat16), mask=mask)

    # Store block total (last valid element's prefix sum)
    last_idx = tl.minimum(block_start + BLOCK_SIZE, N) - 1 - block_start
    block_sum = tl.max(tl.where(tl.arange(0, BLOCK_SIZE) == last_idx, v, tl.zeros_like(v)), axis=0)

    carry_offset = pid_row * num_col_blocks + pid_col
    tl.store(carry_ptr + carry_offset, block_sum)