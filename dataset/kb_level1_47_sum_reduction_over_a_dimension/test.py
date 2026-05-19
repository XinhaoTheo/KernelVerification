import torch
import torch.nn as nn

# Summary: Tests a Triton kernel that performs sum reduction over a specified dimension
# with keepdim=True. The reference model reduces a (128, 4096, 4095) tensor over dim=1,
# producing a (128, 1, 4095) output tensor.

def test_kernel():
    """Test the kernel implementation for sum reduction over a specified dimension."""
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

        # Exact specifications from problem description
        batch_size = 128
        dim1 = 4096
        dim2 = 4095
        reduce_dim = 1

        # Use BF16 as instructed (problem specifies FP32, convert to BF16)
        dtype = torch.bfloat16

        # Create test data using EXACT shapes from problem description
        # Use torch.rand (non-zero) as specified in get_inputs()
        x = torch.rand(batch_size, dim1, dim2, dtype=dtype, device=device)

        # Instantiate reference model using get_init_inputs() spec
        class Model(nn.Module):
            """Simple model that performs sum reduction over a specified dimension."""
            def __init__(self, dim: int):
                super(Model, self).__init__()
                self.dim = dim

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sum(x, dim=self.dim, keepdim=True)

        ref_model = Model(reduce_dim).to(device)

        # Compute reference output
        with torch.no_grad():
            y_ref = ref_model(x)

        print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample (first 5 values): {y_ref.flatten()[:5]}")

        # Call kernel_function as a normal Python function
        # Pass the input tensor and the reduce dimension (not the model)
        result = kernel_function(x, reduce_dim)

        # Verify result is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"ERROR: kernel_function did not return a tensor, got {type(result)}")
            return False

        # Device check (avoid comparing to literal 'cuda')
        if result.device.type != x.device.type:
            print(f"ERROR: result device {result.device} does not match input device {x.device}")
            return False

        print(f"Kernel output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Kernel output sample (first 5 values): {result.flatten()[:5]}")

        # Shape check
        expected_shape = torch.Size([batch_size, 1, dim2])
        if result.shape != expected_shape:
            print(f"ERROR: Shape mismatch. Expected {expected_shape}, got {result.shape}")
            return False

        # Numerical comparison
        # BF16 has lower precision; sum over 4096 elements accumulates error.
        # With 4096 elements summed in BF16, relative error can be significant.
        # Using looser tolerances: rtol=1e-1, atol=1e-1 because:
        # - BF16 has ~7 bits of mantissa (vs 23 for FP32)
        # - Summing 4096 elements in BF16 accumulates substantial rounding error
        # - The reduction dimension is large (4096), amplifying floating point errors
        rtol = 1e-1
        atol = 1e-1

        # Cast result to same dtype as reference for comparison
        result_for_compare = result.to(y_ref.dtype)

        try:
            match = torch.allclose(result_for_compare, y_ref, rtol=rtol, atol=atol)
            if not match:
                abs_diff = torch.abs(result_for_compare - y_ref)
                rel_diff = torch.abs((result_for_compare - y_ref) / (y_ref.abs() + 1e-8))
                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first 10): {y_ref.flatten()[:10]}")
                print(f"Got (first 10): {result_for_compare.flatten()[:10]}")
                print(f"Max absolute difference: {abs_diff.max().item():.6f}")
                print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
                print(f"Max relative error: {rel_diff.max().item():.6f}")
                print(f"Tolerance used: rtol={rtol}, atol={atol}")
                # Additional debug: check a specific slice
                print(f"Sample slice [0, 0, :5] expected: {y_ref[0, 0, :5]}")
                print(f"Sample slice [0, 0, :5] got: {result_for_compare[0, 0, :5]}")
                return False
            else:
                print(f"Numerical check PASSED (rtol={rtol}, atol={atol})")
        except Exception as cmp_e:
            print(f"Error during numerical comparison: {cmp_e}")
            return False

        print("All checks PASSED.")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            import traceback
            print(f"Test failed with exception: {type(e).__name__}: {e}")
            traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)