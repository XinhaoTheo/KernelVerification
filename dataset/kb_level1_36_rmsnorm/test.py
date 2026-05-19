import torch
import torch.nn as nn

# RMS Normalization: normalizes input tensor by dividing by the root mean square
# along the feature dimension (dim=1). Input shape: (batch_size=112, features=64, dim1=512, dim2=512)
# RMS = sqrt(mean(x^2, dim=1, keepdim=True) + eps)
# output = x / RMS

def test_kernel():
    """Test the RMS Normalization kernel implementation."""
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
        batch_size = 112
        features = 64
        dim1 = 512
        dim2 = 512
        eps = 1e-5

        # Use BF16 as per requirements (problem specifies FP32, convert to BF16)
        # Note: Using BF16 requires looser tolerances due to lower precision
        dtype = torch.bfloat16

        # Create test data using EXACT shapes from problem description
        # Use torch.rand (non-zero, positive values) as in get_inputs()
        x = torch.rand(batch_size, features, dim1, dim2, dtype=dtype, device=device)

        print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"Input sample values: {x.flatten()[:5]}")

        # Compute reference output using the PyTorch reference Model
        # Reference Model class (from problem description)
        class Model(nn.Module):
            def __init__(self, num_features: int, eps: float = 1e-5):
                super(Model, self).__init__()
                self.num_features = num_features
                self.eps = eps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
                return x / rms

        # Instantiate reference model and compute reference output
        ref_model = Model(num_features=features, eps=eps).to(device)
        with torch.no_grad():
            y_ref = ref_model(x)

        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample values: {y_ref.flatten()[:5]}")

        # Call kernel_function as a normal Python function
        # Pass the raw input tensor and required parameters (not the model)
        result = kernel_function(x, features, eps)

        print(f"Kernel output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Kernel output sample values: {result.flatten()[:5]}")

        # Device check: verify result is on the same device as input
        if result.device.type != x.device.type:
            print(f"DEVICE MISMATCH: input on {x.device}, result on {result.device}")
            return False

        # Shape check
        if result.shape != x.shape:
            print(f"SHAPE MISMATCH: expected {x.shape}, got {result.shape}")
            return False

        # Numerical comparison
        # Using BF16 tolerances: rtol=1e-2, atol=2e-2
        # BF16 has lower precision (~7 bits mantissa vs 23 for FP32),
        # so we need looser tolerances. Also, the normalization involves
        # mean over 64 features which can accumulate errors.
        rtol = 1e-2
        atol = 2e-2

        # Convert both to float32 for comparison to avoid BF16 rounding in comparison itself
        result_fp32 = result.float()
        y_ref_fp32 = y_ref.float()

        try:
            if not torch.allclose(result_fp32, y_ref_fp32, rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH (rtol={rtol}, atol={atol}):")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first 10): {y_ref_fp32.flatten()[:10]}")
                print(f"Got (first 10): {result_fp32.flatten()[:10]}")
                abs_diff = torch.abs(result_fp32 - y_ref_fp32)
                print(f"Max absolute difference: {torch.max(abs_diff).item()}")
                print(f"Mean absolute difference: {torch.mean(abs_diff).item()}")
                rel_err = torch.abs((result_fp32 - y_ref_fp32) / (y_ref_fp32.abs() + 1e-8))
                print(f"Max relative error: {torch.max(rel_err).item()}")
                # Print where the largest differences occur
                max_idx = torch.argmax(abs_diff)
                print(f"Max diff at flat index {max_idx.item()}: "
                      f"expected={y_ref_fp32.flatten()[max_idx].item():.6f}, "
                      f"got={result_fp32.flatten()[max_idx].item():.6f}")
                return False
            else:
                print(f"Numerical check PASSED (rtol={rtol}, atol={atol})")
        except Exception as e:
            print(f"Error during numerical comparison: {e}")
            return False

        # Additional sanity checks
        # Check that output is not all zeros (would indicate a bug)
        if torch.all(result == 0):
            print("ERROR: Output is all zeros!")
            return False

        # Check that output has finite values
        if not torch.all(torch.isfinite(result)):
            num_nan = torch.sum(torch.isnan(result)).item()
            num_inf = torch.sum(torch.isinf(result)).item()
            print(f"ERROR: Output contains non-finite values: {num_nan} NaNs, {num_inf} Infs")
            return False

        print("All checks passed!")
        return True

    except NameError as e:
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        return False
    except ImportError as e:
        print(f"Test failed: ImportError - could not import kernel_function: {e}")
        return False
    except RuntimeError as e:
        print(f"Test failed: RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"Test failed with unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)