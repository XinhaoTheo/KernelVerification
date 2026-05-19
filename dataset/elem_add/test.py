import torch
import torch.nn as nn

# Element-wise addition of two vectors using a Triton kernel.
# Reference: Model(a, b) = a + b, vector_size=1024, dtype=torch.float32
# Per instructions, we use BF16 instead of FP32 for inputs/outputs.

def test_kernel():
    """Test the Triton kernel for element-wise vector addition."""
    try:
        from kernel import kernel_function

        # Sanity check: kernel should be callable and self-contained
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Device setup
        device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Use EXACT shape from problem description: vector_size = 1024
        # Per instructions, use BF16 instead of FP32
        vector_size = 1024
        dtype = torch.bfloat16  # BF16 as instructed (original was float32)

        # Create test data with non-zero random values
        a = torch.randn(vector_size, dtype=dtype, device=device)
        b = torch.randn(vector_size, dtype=dtype, device=device)

        # Compute reference output using the PyTorch Model from the problem description
        class Model(nn.Module):
            def forward(self, a, b):
                return a + b

        reference_model = Model().to(device)
        y_ref = reference_model(a, b)

        # Call kernel_function as a normal Python function (no grid/launch syntax)
        result = kernel_function(a, b)

        # Verify result is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"ERROR: kernel_function did not return a tensor, got {type(result)}")
            return False

        # Device check: use .device comparison, not string comparison
        if result.device != a.device:
            print(f"ERROR: result device {result.device} does not match input device {a.device}")
            return False

        # Shape check
        if result.shape != a.shape:
            print(f"ERROR: result shape {result.shape} does not match expected shape {a.shape}")
            return False

        # Numerical comparison
        # BF16 has lower precision (~2 decimal digits), so use looser tolerances
        # rtol=1e-2, atol=2e-2 for BF16
        try:
            if not torch.allclose(result.float(), y_ref.float(), rtol=1e-2, atol=2e-2):
                print("NUMERICAL MISMATCH:")
                print(f"  Input a shape: {a.shape}, dtype: {a.dtype}")
                print(f"  Input b shape: {b.shape}, dtype: {b.dtype}")
                print(f"  Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"  Result shape:   {result.shape}, dtype: {result.dtype}")
                print(f"  Input a (first 10): {a[:10]}")
                print(f"  Input b (first 10): {b[:10]}")
                print(f"  Expected (first 10): {y_ref[:10]}")
                print(f"  Got      (first 10): {result[:10]}")
                diff = torch.abs(result.float() - y_ref.float())
                print(f"  Max absolute difference: {diff.max().item():.6f}")
                rel_err = torch.abs((result.float() - y_ref.float()) / (y_ref.float().abs() + 1e-8))
                print(f"  Max relative error:      {rel_err.max().item():.6f}")
                return False
        except Exception as cmp_err:
            print(f"ERROR during numerical comparison: {cmp_err}")
            print(f"  Result: {result}")
            print(f"  Expected: {y_ref}")
            return False

        print("Test PASSED: element-wise addition kernel produces correct results.")
        print(f"  Vector size: {vector_size}, dtype: {dtype}")
        print(f"  Result (first 5): {result[:5]}")
        print(f"  Expected (first 5): {y_ref[:5]}")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test FAILED: NameError (likely undefined helper in kernel.py): {e}")
        elif isinstance(e, RuntimeError):
            print(f"Test FAILED: RuntimeError: {e}")
        else:
            print(f"Test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)