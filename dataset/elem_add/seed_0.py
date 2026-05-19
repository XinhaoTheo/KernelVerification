import triton
import triton.language as tl
import torch


@triton.jit
def _vector_add_kernel(
    ptr_a,       # Pointer to first input vector
    ptr_b,       # Pointer to second input vector
    ptr_out,     # Pointer to output vector
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block (compile-time constant)
):
    """
    Triton kernel for element-wise vector addition: out = a + b
    
    Each program instance handles BLOCK_SIZE elements.
    """
    # Get the program ID for this block
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for all elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundary conditions (last block may be partial)
    mask = offsets < n_elements
    
    # Load elements from both input vectors using Triton
    a = tl.load(ptr_a + offsets, mask=mask, other=0.0)
    b = tl.load(ptr_b + offsets, mask=mask, other=0.0)
    
    # Compute element-wise addition using Triton operations
    result = a + b
    
    # Store result to output vector
    tl.store(ptr_out + offsets, result, mask=mask)


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for element-wise vector addition using a Triton kernel.
    
    Fusion notes:
    - This is a single-stage elementwise operation: load a, load b, add, store.
    - No further fusion is needed since there are no subsequent ops to merge.
    - All computation (load, add, store) happens inside the Triton kernel.
    
    Args:
        a: First input tensor (any shape, bfloat16 or float32)
        b: Second input tensor (same shape and dtype as a)
    
    Returns:
        Output tensor of same shape and dtype as inputs, containing a + b
    """
    # Validate inputs (wrapper-only logic, no compute)
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"
    assert a.device == b.device, f"Device mismatch: {a.device} vs {b.device}"
    assert a.is_cuda, "Input tensors must be on CUDA device"
    
    # Allocate output tensor (PyTorch allocation only, no compute)
    output = torch.empty_like(a)
    
    # Total number of elements to process
    n_elements = a.numel()
    
    # Choose block size (power of 2, good fit for 1024-element vectors)
    BLOCK_SIZE = 1024
    
    # Calculate grid: number of blocks needed to cover all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    _vector_add_kernel[grid](
        a,          # First input pointer
        b,          # Second input pointer
        output,     # Output pointer
        n_elements, # Total elements
        BLOCK_SIZE=BLOCK_SIZE,  # Compile-time block size constant
    )
    
    return output