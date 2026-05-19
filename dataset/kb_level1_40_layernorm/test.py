import torch
import torch.nn as nn

# Summary: Test for a Triton kernel implementing Layer Normalization.
# The model applies nn.LayerNorm with normalized_shape=(features, dim1, dim2)
# to an input tensor of shape (batch_size, features, dim1, dim2) = (16, 64, 256, 256).
# Using BF16 dtype as per requirements (original problem uses FP32).

def test_kernel():
    """Test the Triton kernel implementation of Layer Normalization."""
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
        batch_size = 16
        features = 64
        dim1 = 256
        dim2 = 256
        normalized_shape = (features, dim1, dim2)

        # Create test data using BF16 (converted from FP32 as per requirements)
        # Using torch.rand to match get_inputs() from problem description
        x = torch.rand(batch_size, features, dim1, dim2, device=device, dtype=torch.bfloat16)

        # Instantiate reference model on CUDA using BF16
        # Use get_init_inputs() specification: normalized_shape = (features, dim1, dim2)
        reference_model = nn.LayerNorm(normalized_shape=normalized_shape).to(device=device, dtype=torch.bfloat16)
        reference_model.eval()

        # Compute reference output
        with torch.no_grad():
            y_ref = reference_model(x)

        # Extract weight and bias from the reference model for the kernel call
        # The kernel needs the LayerNorm parameters (weight/gamma and bias/beta)
        weight = reference_model.weight.detach().clone()  # shape: (features, dim1, dim2)
        bias = reference_model.bias.detach().clone()      # shape: (features, dim1, dim2)

        # Call kernel_function as a normal Python function
        # Pass input tensor, weight, bias, and normalized_shape info
        # The kernel should compute LayerNorm internally
        result = kernel_function(x, weight, bias, normalized_shape)

        # Check that result is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"kernel_function did not return a tensor, got: {type(result)}")
            return False

        # Device check: result should be on same device as input
        if result.device.type != x.device.type:
            print(f"Device mismatch: result on {result.device}, input on {x.device}")
            return False

        # Shape check
        if result.shape != x.shape:
            print(f"Shape mismatch: result shape {result.shape}, expected {x.shape}")
            return False

        # Numerical comparison
        # BF16 has lower precision, so use looser tolerances
        # Additionally, LayerNorm normalizes over a large dimension (features*dim1*dim2 = 64*256*256 = 4M elements),
        # which can accumulate significant numerical error in BF16
        rtol = 1e-2
        atol = 2e-2  # BF16 precision requires looser tolerances

        try:
            if not torch.allclose(result.float(), y_ref.float(), rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Weight shape: {weight.shape}, dtype: {weight.dtype}")
                print(f"Bias shape: {bias.shape}, dtype: {bias.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first 10 values): {y_ref.flatten()[:10]}")
                print(f"Got (first 10 values): {result.flatten()[:10]}")
                abs_diff = torch.abs(result.float() - y_ref.float())
                print(f"Max absolute difference: {abs_diff.max().item()}")
                print(f"Mean absolute difference: {abs_diff.mean().item()}")
                rel_err = torch.abs((result.float() - y_ref.float()) / (y_ref.float().abs() + 1e-8))
                print(f"Max relative error: {rel_err.max().item()}")
                print(f"Input sample values: {x.flatten()[:10]}")
                # Additional debug: check a specific batch element
                print(f"Result batch 0, feature 0, first 5: {result[0, 0, 0, :5]}")
                print(f"Expected batch 0, feature 0, first 5: {y_ref[0, 0, 0, :5]}")
                return False
        except Exception as compare_err:
            print(f"Error during numerical comparison: {compare_err}")
            print(f"Result dtype: {result.dtype}, y_ref dtype: {y_ref.dtype}")
            print(f"Result shape: {result.shape}, y_ref shape: {y_ref.shape}")
            return False

        print(f"Test PASSED!")
        print(f"Input shape: {x.shape}, dtype: {x.dtype}")
        print(f"Output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Max absolute difference from reference: {torch.abs(result.float() - y_ref.float()).max().item():.6f}")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        elif isinstance(e, ImportError):
            print(f"Test failed: ImportError - could not import kernel_function: {e}")
        elif isinstance(e, RuntimeError):
            print(f"Test failed: RuntimeError: {e}")
        else:
            print(f"Test failed with {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)