import torch

# Row-wise softmax along the last dimension (dim=-1)
# Model: torch.softmax(x, dim=-1)
# Exact shapes: batch=32, dim=1024, dtype=torch.float32 (tested as bfloat16 per requirements)

def test_kernel():
    """Test the Triton kernel implementation of row-wise softmax."""
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

        # Use exact shapes from problem description
        # Problem specifies FP32, but per requirements we use BF16 instead
        batch = 32
        dim = 1024
        dtype = torch.bfloat16  # Using BF16 as per requirements (problem specifies FP32)

        # Create test input using exact specifications
        # Using torch.randn for non-zero random data
        x = torch.randn(batch, dim, dtype=dtype, device=device)

        print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"Input sample (first row, first 5 values): {x[0, :5]}")

        # Call kernel_function as a normal Python function
        result = kernel_function(x)

        # Device check
        if isinstance(result, torch.Tensor):
            if result.device.type != x.device.type:
                print(f"Device mismatch: result device {result.device} vs input device {x.device}")
                return False
        else:
            print(f"kernel_function did not return a tensor, got: {type(result)}")
            return False

        print(f"Result shape: {result.shape}, dtype: {result.dtype}")

        # Compute reference output using PyTorch reference Model
        # Instantiate reference model as specified in problem description
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)

        ref_model = Model().to(device)
        with torch.no_grad():
            y_ref = ref_model(x)

        print(f"Reference shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference sample (first row, first 5 values): {y_ref[0, :5]}")
        print(f"Result sample (first row, first 5 values): {result[0, :5]}")

        # Verify softmax properties: values sum to 1 per row and are in [0, 1]
        row_sums = result.sum(dim=-1)
        print(f"Row sums (should all be ~1.0): min={row_sums.min().item():.6f}, max={row_sums.max().item():.6f}")

        # Check that all values are in valid probability range
        if not (result >= 0).all():
            print("FAILED: Some softmax output values are negative!")
            print(f"Min value: {result.min().item()}")
            return False

        if not (result <= 1).all():
            print("FAILED: Some softmax output values exceed 1!")
            print(f"Max value: {result.max().item()}")
            return False

        # Numerical comparison against reference
        # BF16 has lower precision, so we use looser tolerances
        # rtol=1e-2, atol=2e-2 for BF16 (lower precision than FP32)
        try:
            if not torch.allclose(result.float(), y_ref.float(), rtol=1e-2, atol=2e-2):
                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first row, first 10): {y_ref[0, :10]}")
                print(f"Got (first row, first 10):      {result[0, :10]}")
                abs_diff = torch.abs(result.float() - y_ref.float())
                rel_err = torch.abs((result.float() - y_ref.float()) / (y_ref.float() + 1e-8))
                print(f"Max absolute difference: {abs_diff.max().item():.6f}")
                print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
                print(f"Max relative error: {rel_err.max().item():.6f}")
                # Show where the worst differences are
                worst_idx = abs_diff.argmax()
                row_idx = worst_idx // dim
                col_idx = worst_idx % dim
                print(f"Worst difference at row={row_idx.item()}, col={col_idx.item()}: "
                      f"expected={y_ref[row_idx, col_idx].item():.6f}, "
                      f"got={result[row_idx, col_idx].item():.6f}")
                return False
        except Exception as e:
            print(f"Error during numerical comparison: {e}")
            return False

        # Verify row sums are close to 1.0
        ones = torch.ones(batch, dtype=torch.float32, device=device)
        if not torch.allclose(row_sums.float(), ones, rtol=1e-2, atol=2e-2):
            print(f"FAILED: Row sums are not close to 1.0")
            print(f"Row sums: {row_sums}")
            return False

        print("All checks passed!")
        print(f"  - Output shape correct: {result.shape == x.shape}")
        print(f"  - All values in [0, 1]: True")
        print(f"  - Row sums close to 1.0: True")
        print(f"  - Numerical match with torch.softmax: True (rtol=1e-2, atol=2e-2 for BF16)")

        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        elif isinstance(e, ImportError):
            print(f"Test failed: ImportError - could not import kernel_function: {e}")
        else:
            import traceback
            print(f"Test failed with exception: {type(e).__name__}: {e}")
            traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    if success:
        print("\n✓ Test PASSED")
        sys.exit(0)
    else:
        print("\n✗ Test FAILED")
        sys.exit(1)