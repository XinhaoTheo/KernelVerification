import triton
import triton.language as tl
import torch

# Fused operation: Single-pass sigmoid activation
# sigmoid(x) = 1 / (1 + exp(-x))
# This kernel loads input values, computes sigmoid in-place using Triton math ops,
# and stores the result. No separate passes needed — fully fused single kernel.

@triton.jit
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise sigmoid activation.
    
    Computes: out = 1 / (1 + exp(-x))
    
    Each program instance handles a contiguous block of BLOCK_SIZE elements.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_x = -x_f32
    exp_neg_x = tl.exp(neg_x)
    sigmoid_val = 1.0 / (1.0 + exp_neg_x)

    # Cast back to original dtype and store
    out = sigmoid_val.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Triton sigmoid kernel.
    
    Fused stages (single kernel pass):
      1. Load input element
      2. Compute sigmoid: 1 / (1 + exp(-x))  [in fp32 for stability]
      3. Cast result back to input dtype (bfloat16)
      4. Store output
    
    Args:
        x: Input tensor of any shape, dtype bfloat16 (or float32)
    
    Returns:
        Output tensor of same shape and dtype as input, with sigmoid applied.
    """
    # Validate input
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Flatten to 1D for simple indexing
    n_elements = x.numel()
    
    # Choose block size: 1024 is a good default for elementwise ops
    BLOCK_SIZE = 1024
    
    # Grid: one program per block of BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch Triton kernel
    _sigmoid_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out