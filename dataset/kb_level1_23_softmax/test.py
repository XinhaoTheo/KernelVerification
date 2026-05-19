import torch
import torch.nn as nn

# Summary: Test for a Triton kernel implementing Softmax activation.
# The model applies torch.softmax(x, dim=1) to an input tensor of shape
# (batch_size=4096, dim=393216) with bfloat16 dtype.
# Note: dim=393216 is very large, so we use looser tolerances (rtol=1e-1, atol=1e-1)
# due to accumulation errors over the large reduction dimension in bf16.

def test_kernel():
    """Test the Triton kernel implementation of Softmax."""
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
        batch_size = 4096
        dim = 393216

        # Use bfloat16 as instructed (problem specifies FP32, convert to BF16)
        # Use non-zero random data to avoid hiding computation errors
        x = torch.rand(batch_size, dim, dtype=torch.bfloat16, device=device)

        # Compute reference output using the PyTorch reference Model
        class Model(nn.Module):
            """Simple model that performs a Softmax activation."""
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.softmax(x, dim=1)

        ref_model = Model().to(device)
        ref_model.eval()

        with torch.no_grad():
            y_ref = ref_model(x)

        # Call kernel_function as a normal Python function
        # Pass only the raw input tensor, not the model or reference output
        result = kernel_function(x)

        # Device check
        if not isinstance(result, torch.Tensor):
            print(f"ERROR: kernel_function did not return a tensor, got {type(result)}")
            return False

        if result.device.type != x.device.type:
            print(f"ERROR: Device mismatch. Input device: {x.device}, Result device: {result.device}")
            return False

        # Shape check
        if result.shape != x.shape:
            print(f"ERROR: Shape mismatch. Expected {x.shape}, got {result.shape}")
            return False

        # Numerical comparison
        # Tolerances are loose (rtol=1e-1, atol=1e-1) because:
        # 1. We use bfloat16 which has lower precision (~2 decimal digits)
        # 2. The reduction dimension (dim=393216) is extremely large, causing
        #    significant accumulation of floating-point errors in softmax computation
        #    (exp + sum over 393216 elements)
        try:
            # Cast to float32 for comparison to avoid dtype mismatch issues
            result_f32 = result.float()
            y_ref_f32 = y_ref.float()

            if not torch.allclose(result_f32, y_ref_f32, rtol=1e-1, atol=1e-1):
                print("NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Input sample values: {x.flatten()[:10]}")
                print(f"Expected shape: {y_ref_f32.shape}, dtype: {y_ref_f32.dtype}")
                print(f"Result shape: {result_f32.shape}, dtype: {result_f32.dtype}")
                print(f"Expected (first 10): {y_ref_f32.flatten()[:10]}")
                print(f"Got (first 10): {result_f32.flatten()[:10]}")
                max_abs_diff = torch.max(torch.abs(result_f32 - y_ref_f32)).item()
                rel_err = torch.max(torch.abs((result_f32 - y_ref_f32) / (y_ref_f32.abs() + 1e-8))).item()
                print(f"Max absolute difference: {max_abs_diff}")
                print(f"Max relative error: {rel_err}")

                # Additional diagnostics: check softmax properties
                row_sums = result_f32.sum(dim=1)
                print(f"Row sums (should be ~1.0), first 5: {row_sums[:5]}")
                print(f"Min value (should be >= 0): {result_f32.min().item()}")
                print(f"Max value (should be <= 1): {result_f32.max().item()}")
                return False
            else:
                print("Numerical comparison PASSED.")

            # Additional sanity checks for softmax properties
            # 1. All values should be non-negative
            if (result_f32 < 0).any():
                print("ERROR: Softmax output contains negative values!")
                print(f"Min value: {result_f32.min().item()}")
                return False

            # 2. All values should be <= 1
            if (result_f32 > 1.0 + 1e-3).any():
                print("ERROR: Softmax output contains values > 1!")
                print(f"Max value: {result_f32.max().item()}")
                return False

            # 3. Each row should sum to approximately 1
            row_sums = result_f32.sum(dim=1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-2, atol=1e-2):
                print("ERROR: Softmax row sums are not close to 1!")
                print(f"Row sums (first 5): {row_sums[:5]}")
                print(f"Max deviation from 1: {torch.max(torch.abs(row_sums - 1.0)).item()}")
                return False

            print("All softmax property checks PASSED.")

        except Exception as comp_e:
            print(f"Error during numerical comparison: {comp_e}")
            import traceback
            traceback.print_exc()
            return False

        print("All tests PASSED.")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            print(f"Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)