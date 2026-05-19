import triton
import triton.language as tl
import torch


@triton.jit
def _rms_norm_kernel(
    x_ptr,          # Input tensor pointer
    out_ptr,        # Output tensor pointer
    features,       # Number of features (reduction dim, dim=1)
    dim1,           # Size of spatial dim1
    dim2,           # Size of spatial dim2
    eps,            # Epsilon for numerical stability
    stride_b,       # Stride for batch dimension
    stride_f,       # Stride for feature dimension
    stride_d1,      # Stride for dim1
    stride_d2,      # Stride for dim2
    BLOCK_F: tl.constexpr,  # Block size for feature dimension (must cover all features)
):
    """
    Fused RMS Normalization kernel.
    
    Fused stages (single pass):
      Stage 1: Load x along feature dim, compute x^2
      Stage 2: Reduce sum(x^2) over features -> mean -> RMS
      Stage 3: Normalize: x / RMS
      Stage 4: Store result
    
    Each program handles one (batch, dim1, dim2) spatial position.
    """
    # Each program handles one spatial position: (b, :, d1, d2)
    pid = tl.program_id(0)
    
    # Total number of spatial positions = batch_size * dim1 * dim2
    # Decode pid -> (b, d1, d2)
    d2_idx = pid % dim2
    tmp = pid // dim2
    d1_idx = tmp % dim1
    b_idx = tmp // dim1
    
    # Base pointer for this spatial position
    base_offset = b_idx * stride_b + d1_idx * stride_d1 + d2_idx * stride_d2
    
    # Feature offsets
    f_offs = tl.arange(0, BLOCK_F)
    mask = f_offs < features
    
    # Load all features for this spatial position
    x_ptrs = x_ptr + base_offset + f_offs * stride_f
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sum of squares (fused: square + reduce)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute mean and RMS
    mean_sq = sum_sq / features
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize
    x_norm = x / rms
    
    # Cast back to original dtype and store
    out_ptrs = out_ptr + base_offset + f_offs * stride_f
    tl.store(out_ptrs, x_norm.to(out_ptr.dtype.element_ty), mask=mask)


def kernel_function(x: torch.Tensor, features: int, eps: float = 1e-5) -> torch.Tensor:
    """
    RMS Normalization wrapper.
    
    Computes: output = x / sqrt(mean(x^2, dim=1, keepdim=True) + eps)
    
    Fused kernel covers:
      - Squaring elements along feature dim
      - Summing and averaging (mean of squares)
      - Square root + epsilon
      - Division (normalization)
    All in a single Triton kernel pass, avoiding intermediate tensor allocations.
    
    Args:
        x: Input tensor of shape (batch_size, features, dim1, dim2)
        features: Number of features (size of dim=1)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of same shape and dtype as input
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    assert x.shape[1] == features, f"features mismatch: tensor has {x.shape[1]}, got {features}"
    
    batch_size, feat, dim1, dim2 = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Each program handles one (batch, dim1, dim2) position
    total_positions = batch_size * dim1 * dim2
    
    # BLOCK_F must be a power of 2 >= features
    # features=64 -> BLOCK_F=64
    BLOCK_F = triton.next_power_of_2(features)
    
    # Grid: one program per spatial position
    grid = (total_positions,)
    
    _rms_norm_kernel[grid](
        x,
        out,
        features,
        dim1,
        dim2,
        eps,
        x.stride(0),  # stride_b
        x.stride(1),  # stride_f
        x.stride(2),  # stride_d1
        x.stride(3),  # stride_d2
        BLOCK_F=BLOCK_F,
    )
    
    return out