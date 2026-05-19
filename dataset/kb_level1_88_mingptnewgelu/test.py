import torch
import torch.nn as nn
import math

# Test for GELU activation function kernel
# Based on the implementation from minGPT (Gaussian Error Linear Units)
# Formula: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# Input shape: (8192, 8192), using BF16 (converted from FP32 as per requirements)

def test_kernel():
    """Test the GELU kernel implementation against PyTorch reference."""
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

        # Reference Model from problem description
        class Model(nn.Module):
            """
            Implementation of the GELU activation function currently in Google BERT repo
            (identical to OpenAI GPT).
            Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
            """
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

        # Use EXACT shapes from problem description
        batch_size = 8192
        dim = 8192

        # Create input tensor using torch.rand (as in get_inputs())
        # Using BF16 as per requirements (converted from FP32)
        # Using rand (values in [0, 1]) to match get_inputs()
        x_fp32 = torch.rand(batch_size, dim, device=device)
        x_bf16 = x_fp32.to(torch.bfloat16)

        print(f"Input shape: {x_bf16.shape}, dtype: {x_bf16.dtype}, device: {x_bf16.device}")
        print(f"Input sample values: {x_bf16.flatten()[:5]}")

        # Compute reference output using the PyTorch Model
        model = Model().to(device)
        model.eval()
        with torch.no_grad():
            y_ref = model(x_bf16)

        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample values: {y_ref.flatten()[:5]}")

        # Call kernel_function as a normal Python function
        # Pass only the input tensor - no reference outputs, no model
        result = kernel_function(x_bf16)

        print(f"Kernel output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Kernel output sample values: {result.flatten()[:5]}")

        # Device check (avoid comparing to literal 'cuda')
        if isinstance(result, torch.Tensor) and result.device.type != x_bf16.device.type:
            print(f"Device mismatch: result on {result.device}, expected {x_bf16.device}")
            return False

        # Shape check
        if result.shape != y_ref.shape:
            print(f"Shape mismatch: result {result.shape}, expected {y_ref.shape}")
            return False

        # Numerical comparison
        # Using BF16 tolerances: rtol=1e-2, atol=2e-2 because BF16 has lower precision
        # (only ~3 decimal digits of precision vs ~7 for FP32)
        rtol = 1e-2
        atol = 2e-2

        try:
            # Cast both to float32 for comparison to avoid BF16 comparison issues
            result_fp32 = result.float()
            y_ref_fp32 = y_ref.float()

            if not torch.allclose(result_fp32, y_ref_fp32, rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x_bf16.shape}, dtype: {x_bf16.dtype}")
                print(f"Expected shape: {y_ref_fp32.shape}, dtype: {y_ref_fp32.dtype}")
                print(f"Result shape: {result_fp32.shape}, dtype: {result_fp32.dtype}")
                print(f"Expected (first 10): {y_ref_fp32.flatten()[:10]}")
                print(f"Got (first 10): {result_fp32.flatten()[:10]}")
                abs_diff = torch.abs(result_fp32 - y_ref_fp32)
                print(f"Max absolute difference: {torch.max(abs_diff).item()}")
                print(f"Mean absolute difference: {torch.mean(abs_diff).item()}")
                rel_diff = torch.abs((result_fp32 - y_ref_fp32) / (y_ref_fp32.abs() + 1e-8))
                print(f"Max relative error: {torch.max(rel_diff).item()}")
                # Show where the biggest differences are
                max_idx = torch.argmax(abs_diff)
                print(f"Largest diff at flat index {max_idx.item()}: "
                      f"expected={y_ref_fp32.flatten()[max_idx].item():.6f}, "
                      f"got={result_fp32.flatten()[max_idx].item():.6f}")
                return False

            print(f"Numerical check PASSED (rtol={rtol}, atol={atol})")

        except Exception as e:
            print(f"Error during numerical comparison: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Additional sanity checks
        # GELU output should be in a reasonable range for inputs in [0, 1]
        # For x in [0, 1], GELU(x) should be in [0, ~1]
        if torch.any(torch.isnan(result)):
            print("ERROR: Result contains NaN values")
            return False

        if torch.any(torch.isinf(result)):
            print("ERROR: Result contains Inf values")
            return False

        print("All checks passed!")
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