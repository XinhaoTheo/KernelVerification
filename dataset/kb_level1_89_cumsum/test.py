import torch
import torch.nn as nn

# Cumulative sum (prefix sum) along a specified dimension.
# Reference model: torch.cumsum(x, dim=self.dim)
# Input shape: (32768, 32768), dim=1, dtype=bfloat16 (converted from FP32 per instructions)
# Note: Large accumulation dimension (32768) may introduce significant floating point error,
# so we use loose tolerances.

def test_kernel():
    """Test the cumulative sum kernel implementation."""
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
        batch_size = 32768
        input_shape = (32768,)
        dim = 1

        # Use bfloat16 as per instructions (problem specifies FP32, convert to BF16)
        # torch.rand generates values in [0, 1), which avoids all-zero issues
        x = torch.rand(batch_size, *input_shape, dtype=torch.bfloat16, device=device)

        # Reference model computation
        class Model(nn.Module):
            def __init__(self, dim):
                super(Model, self).__init__()
                self.dim = dim

            def forward(self, x):
                return torch.cumsum(x, dim=self.dim)

        ref_model = Model(dim=dim).to(device)
        y_ref = ref_model(x)

        # Call kernel_function as a normal Python function
        # Pass the input tensor and dim as arguments (kernel decides its own API)
        result = kernel_function(x, dim)

        # Device check
        if isinstance(result, torch.Tensor):
            if result.device.type != x.device.type:
                print(f"Device mismatch: result on {result.device}, input on {x.device}")
                return False
        else:
            print(f"kernel_function did not return a tensor, got: {type(result)}")
            return False

        # Shape check
        if result.shape != y_ref.shape:
            print(f"Shape mismatch: expected {y_ref.shape}, got {result.shape}")
            return False

        # Numerical comparison
        # Tolerances are loose because:
        # 1. bfloat16 has limited precision (rtol=1e-2, atol=2e-2 baseline for bf16)
        # 2. The accumulation dimension is very large (32768 elements), causing significant
        #    floating point error accumulation. Using much larger tolerances as instructed.
        # 3. Values in [0, 1) accumulated over 32768 elements can reach ~16384, so
        #    absolute error can be substantial.
        rtol = 1e-1
        atol = 1e-1  # Large due to massive accumulation dimension (32768)

        try:
            if not torch.allclose(result.float(), y_ref.float(), rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first few): {y_ref.flatten()[:10]}")
                print(f"Got (first few): {result.flatten()[:10]}")
                print(f"Expected (last few along dim=1 for row 0): {y_ref[0, -10:]}")
                print(f"Got (last few along dim=1 for row 0): {result[0, -10:]}")
                abs_diff = torch.abs(result.float() - y_ref.float())
                rel_diff = torch.abs((result.float() - y_ref.float()) / (y_ref.float().abs() + 1e-8))
                print(f"Max absolute difference: {abs_diff.max().item()}")
                print(f"Mean absolute difference: {abs_diff.mean().item()}")
                print(f"Max relative error: {rel_diff.max().item()}")
                print(f"Mean relative error: {rel_diff.mean().item()}")
                # Show where the worst errors are
                worst_idx = abs_diff.argmax()
                worst_row = worst_idx // x.shape[1]
                worst_col = worst_idx % x.shape[1]
                print(f"Worst error at row={worst_row}, col={worst_col}: "
                      f"expected={y_ref[worst_row, worst_col].item():.4f}, "
                      f"got={result[worst_row, worst_col].item():.4f}")
                return False
        except Exception as cmp_e:
            print(f"Comparison failed with exception: {cmp_e}")
            print(f"Result dtype: {result.dtype}, y_ref dtype: {y_ref.dtype}")
            print(f"Result shape: {result.shape}, y_ref shape: {y_ref.shape}")
            return False

        print(f"Test PASSED!")
        print(f"Input shape: {x.shape}, dtype: {x.dtype}")
        print(f"Output shape: {result.shape}, dtype: {result.dtype}")
        abs_diff = torch.abs(result.float() - y_ref.float())
        print(f"Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed with exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)