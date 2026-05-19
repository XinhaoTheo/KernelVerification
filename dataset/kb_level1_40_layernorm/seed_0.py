import triton
import triton.language as tl
import torch

@triton.jit
def _layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Pass 1: compute mean and variance using online Welford or two-pass
    mean = tl.zeros((), dtype=tl.float32)
    # First accumulate sum for mean
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / N

    # Second pass: compute variance
    var_val = tl.zeros((), dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var_val += tl.sum(diff * diff, axis=0)
    var_val = var_val / N
    rstd = 1.0 / tl.sqrt(var_val + eps)

    # Third pass: normalize and apply affine transform
    for off in tl.range(0, N, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * rstd
        out = x_norm * w + b
        tl.store(out_row_ptr + offs, out.to(tl.bfloat16), mask=mask)


def kernel_function(x, weight, bias, normalized_shape):
    """
    Fused LayerNorm kernel.
    
    Fused stages (all in one kernel):
      1. Pass 1: Accumulate sum → mean (fp32 accumulation)
      2. Pass 2: Accumulate squared deviations → variance → rstd
      3. Pass 3: Normalize + affine scale+shift → store as bf16
    
    Each Triton program handles one batch element (row), iterating over
    N = prod(normalized_shape) elements in BLOCK_SIZE chunks.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    
    # Determine batch size and normalization size
    N = 1
    for s in normalized_shape:
        N *= s
    batch_size = x.numel() // N
    
    # Flatten for kernel access
    x_flat = x.contiguous().view(batch_size, N)
    w_flat = weight.contiguous().view(N)
    b_flat = bias.contiguous().view(N)
    out_flat = torch.empty_like(x_flat)
    
    eps = 1e-5
    BLOCK_SIZE = 1024  # Process 1024 elements per iteration
    
    grid = (batch_size,)
    _layernorm_kernel[grid](
        x_flat, w_flat, b_flat, out_flat,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_flat.view(x.shape)