import torch
import torch.nn as nn

# Test for a simple ReLU activation Triton kernel.
# The model applies torch.relu(x) to an input tensor of shape (4096, 393216).
# We use BF16 instead of FP32 as per requirements.

def test_kernel():
    """Test the ReLU kernel implementation."""
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
        # torch.rand gives values in [0, 1), but we want to test negative values too
        # so we shift to [-0.5, 0.5) to ensure both positive and negative values
        x = torch.rand(batch_size, dim, dtype=torch.bfloat16, device=device) - 0.5

        print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"Input sample values: {x.flatten()[:10]}")
        print(f"Input min: {x.min().item():.4f}, max: {x.max().item():.4f}")

        # Compute reference output using the PyTorch Model from problem description
        class Model(nn.Module):
            """Simple model that performs a ReLU activation."""
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x)

        ref_model = Model().to(device)
        ref_model.eval()

        with torch.no_grad():
            y_ref = ref_model(x)

        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample: {y_ref.flatten()[:10]}")

        # Call kernel_function as a normal Python function
        result = kernel_function(x)

        # Verify result is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"ERROR: kernel_function did not return a tensor, got {type(result)}")
            return False

        # Device check: use device.type comparison to avoid literal 'cuda' string comparison
        if result.device.type != x.device.type:
            print(f"ERROR: Result device {result.device} does not match input device {x.device}")
            return False

        # Shape check
        if result.shape != x.shape:
            print(f"ERROR: Shape mismatch. Expected {x.shape}, got {result.shape}")
            return False

        print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        print(f"Result sample values: {result.flatten()[:10]}")

        # Verify no negative values (ReLU property)
        if (result < 0).any():
            neg_count = (result < 0).sum().item()
            print(f"ERROR: Result contains {neg_count} negative values (ReLU should clamp to 0)")
            print(f"Negative values sample: {result[result < 0][:10]}")
            return False

        # Numerical comparison against reference
        # BF16 has lower precision, use slightly relaxed tolerances
        # rtol=1e-2, atol=2e-2 for bfloat16 as documented
        try:
            # Cast to float32 for comparison to avoid dtype mismatch issues
            result_f32 = result.float()
            y_ref_f32 = y_ref.float()

            if not torch.allclose(result_f32, y_ref_f32, rtol=1e-2, atol=2e-2):
                max_abs_diff = torch.max(torch.abs(result_f32 - y_ref_f32)).item()
                # Avoid division by zero
                rel_err = torch.max(torch.abs((result_f32 - y_ref_f32) / (y_ref_f32.abs() + 1e-8))).item()

                print(f"NUMERICAL MISMATCH:")
                print(f"Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"Expected shape: {y_ref_f32.shape}, dtype: {y_ref_f32.dtype}")
                print(f"Result shape: {result_f32.shape}, dtype: {result_f32.dtype}")
                print(f"Expected (first 10): {y_ref_f32.flatten()[:10]}")
                print(f"Got (first 10):      {result_f32.flatten()[:10]}")
                print(f"Max absolute difference: {max_abs_diff}")
                print(f"Max relative error: {rel_err}")

                # Find where mismatches occur
                mismatch_mask = ~torch.isclose(result_f32, y_ref_f32, rtol=1e-2, atol=2e-2)
                mismatch_count = mismatch_mask.sum().item()
                total = result_f32.numel()
                print(f"Number of mismatched elements: {mismatch_count} / {total} ({100.0 * mismatch_count / total:.4f}%)")

                if mismatch_count > 0:
                    mismatch_indices = mismatch_mask.nonzero(as_tuple=False)[:5]
                    print(f"First few mismatch locations: {mismatch_indices.tolist()}")
                    for idx in mismatch_indices[:5]:
                        i, j = idx[0].item(), idx[1].item()
                        print(f"  [{i},{j}]: input={x[i,j].item():.6f}, expected={y_ref_f32[i,j].item():.6f}, got={result_f32[i,j].item():.6f}")

                return False

        except Exception as cmp_e:
            print(f"Error during numerical comparison: {cmp_e}")
            return False

        print("All checks passed!")
        print(f"ReLU kernel test PASSED with BF16 input of shape {x.shape}")
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