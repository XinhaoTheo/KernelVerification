import torch
import torch.nn as nn

# Summary: Test for a Triton kernel implementing Sigmoid activation.
# The model applies torch.sigmoid() to an input tensor of shape (4096, 393216).
# We use BF16 instead of FP32 as per requirements.

def test_kernel():
    """Test the Triton kernel implementation of Sigmoid activation."""
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

        # Use EXACT shapes from problem description
        batch_size = 4096
        dim = 393216

        # Use BF16 instead of FP32 as per requirements
        # torch.rand generates values in [0, 1), which is non-zero
        x = torch.rand(batch_size, dim, dtype=torch.bfloat16, device=device)

        print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"Input sample values: {x.flatten()[:10]}")

        # Reference model: instantiate the PyTorch reference Model
        class Model(nn.Module):
            """Simple model that performs a Sigmoid activation."""
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(x)

        ref_model = Model().to(device)
        ref_model.eval()

        # Compute reference output (kept local, not passed to kernel_function)
        with torch.no_grad():
            y_ref = ref_model(x)

        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample values: {y_ref.flatten()[:10]}")

        # Call kernel_function as a normal Python function
        # Pass only the input tensor (same as what the model receives)
        result = kernel_function(x)

        # Verify result is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"ERROR: kernel_function did not return a tensor, got {type(result)}")
            return False

        # Device check: use device comparison, not string comparison
        if result.device.type != x.device.type:
            print(f"ERROR: result device {result.device} does not match input device {x.device}")
            return False

        print(f"Kernel output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Kernel output sample values: {result.flatten()[:10]}")

        # Shape check
        if result.shape != x.shape:
            print(f"ERROR: Shape mismatch. Expected {x.shape}, got {result.shape}")
            return False

        # Numerical comparison
        # BF16 has lower precision, so use slightly relaxed tolerances
        # rtol=1e-2, atol=2e-2 because BF16 has ~7 bits of mantissa precision
        try:
            # Cast both to float32 for comparison to avoid BF16 precision issues in comparison itself
            result_fp32 = result.float()
            y_ref_fp32 = y_ref.float()

            if not torch.allclose(result_fp32, y_ref_fp32, rtol=1e-2, atol=2e-2):
                max_abs_diff = torch.max(torch.abs(result_fp32 - y_ref_fp32))
                rel_err = torch.max(torch.abs((result_fp32 - y_ref_fp32) / (y_ref_fp32.abs() + 1e-8)))
                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first 10): {y_ref_fp32.flatten()[:10]}")
                print(f"Got (first 10): {result_fp32.flatten()[:10]}")
                print(f"Max absolute difference: {max_abs_diff.item()}")
                print(f"Max relative error: {rel_err.item()}")
                # Check how many elements are mismatched
                mismatched = ~torch.isclose(result_fp32, y_ref_fp32, rtol=1e-2, atol=2e-2)
                print(f"Number of mismatched elements: {mismatched.sum().item()} / {result.numel()}")
                return False
            else:
                max_abs_diff = torch.max(torch.abs(result_fp32 - y_ref_fp32))
                print(f"Numerical check PASSED. Max absolute difference: {max_abs_diff.item():.6f}")

        except Exception as e:
            print(f"Error during numerical comparison: {e}")
            return False

        # Additional sanity checks: sigmoid output should be in (0, 1)
        try:
            result_fp32 = result.float()
            if torch.any(result_fp32 < 0) or torch.any(result_fp32 > 1):
                print(f"ERROR: Sigmoid output contains values outside [0, 1]")
                print(f"Min value: {result_fp32.min().item()}, Max value: {result_fp32.max().item()}")
                return False
            print(f"Range check PASSED: output in [{result_fp32.min().item():.6f}, {result_fp32.max().item():.6f}]")
        except Exception as e:
            print(f"Error during range check: {e}")
            return False

        print("All checks PASSED!")
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
        print(f"Test failed with unexpected exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)