import torch
import triton
import triton.language as tl


# Fused stages: This kernel implements a single-pass matrix multiplication C = A * B
# using tl.dot for tensor core acceleration. The entire computation (load tiles of A and B,
# accumulate dot products, store result) is fused into one Triton kernel with no separate passes.


def get_matmul_configs():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK, 'GROUP_SIZE_M': 8},
            num_stages=s, num_warps=w
        )
        for BM in [64, 128]
        for BN in [64, 128]
        for BK in [32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]


@triton.autotune(
    configs=get_matmul_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A * B.

    Fused stages:
      1. Tile loading: Load BLOCK_SIZE_M x BLOCK_SIZE_K tiles of A and
         BLOCK_SIZE_K x BLOCK_SIZE_N tiles of B from global memory.
      2. Dot product accumulation: Use tl.dot to accumulate partial results
         into a float32 accumulator (for numerical stability with bfloat16 inputs).
      3. Store: Write the final BLOCK_SIZE_M x BLOCK_SIZE_N tile of C back
         to global memory, cast to the output dtype.

    All three stages run in a single kernel pass, minimizing memory traffic.
    """
    # Program IDs
    pid = tl.program_id(axis=0)

    # Number of tiles along M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grouped ordering for better L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Starting offsets for this tile
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Clamp offsets to valid range (for masking)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Pointers to the first tiles of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator in float32 for precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main K-loop: load tiles and accumulate dot products
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Boundary mask for K dimension
        k_remaining = K - k * BLOCK_SIZE_K
        mask_k = offs_k < k_remaining

        # Load tiles with boundary masking
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # Accumulate: fused tile multiply-add using tensor cores
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers along K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Compute output indices
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Boundary mask for output
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Pointers to output tile
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Store result, cast to output dtype (bfloat16 to match input)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Triton matrix multiplication kernel: C = A * B.

    Fusion note:
      A single Triton kernel (_matmul_kernel) fuses:
        - Tiled loading of A and B with boundary masking
        - Float32 accumulation via tl.dot (tensor cores)
        - BF16 store of the output tile
      No intermediate tensors are materialized; the entire computation
      runs in one kernel launch.

    Args:
        A: Left matrix of shape (M, K), dtype bfloat16, on CUDA.
        B: Right matrix of shape (K, N), dtype bfloat16, on CUDA.

    Returns:
        C: Output matrix of shape (M, N), dtype bfloat16, on CUDA.
    """
    # Validate inputs
    assert A.ndim == 2 and B.ndim == 2, "Both inputs must be 2D matrices"
    assert A.shape[1] == B.shape[0], (
        f"Incompatible dimensions: A.shape={A.shape}, B.shape={B.shape}"
    )
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    assert A.dtype == B.dtype, f"dtype mismatch: A={A.dtype}, B={B.dtype}"

    M, K = A.shape
    K2, N = B.shape

    # Allocate output tensor (same dtype as inputs)
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Ensure tensors are contiguous for optimal memory access
    A = A.contiguous()
    B = B.contiguous()

    # Grid: one program per output tile
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the fused Triton matmul kernel
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C