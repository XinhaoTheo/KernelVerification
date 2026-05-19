import torch
import torch.nn as nn

# Test for a Triton kernel implementing GELU activation.
# Reference: torch.nn.functional.gelu applied to a tensor of shape (4096, 393216).
# The model simply applies GELU elementwise.

def test_kernel():
    """Test the GELU kernel implementation."""
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

        # Use EXACT shapes from problem description:
        # batch_size = 4096, dim = 393216
        # Problem specifies FP32 (via torch.rand), but per requirements we use BF16 instead.
        batch_size = 4096
        dim = 393216

        print(f"Creating input tensor of shape ({batch_size}, {dim}) with dtype=bfloat16...")

        # Use non-zero random input (torch.rand gives values in [0,1), but GELU is
        # most interesting over a wider range; use randn for better coverage)
        x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device=device)

        print(f"Input tensor created: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        print(f"Input sample (first 10): {x.flatten()[:10]}")

        # Compute reference output using the PyTorch reference model
        class Model(nn.Module):
            """Simple model that performs a GELU activation."""
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x)

        reference_model = Model().to(device)
        reference_model.eval()

        with torch.no_grad():
            y_ref = reference_model(x)

        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample (first 10): {y_ref.flatten()[:10]}")

        # Call kernel_function as a normal Python function
        # The kernel receives only the input tensor (no reference outputs)
        result = kernel_function(x)

        print(f"Kernel output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Kernel output sample (first 10): {result.flatten()[:10]}")

        # Device check: result should be on the same device as input
        if result.device.type != x.device.type:
            print(f"DEVICE MISMATCH: result.device={result.device}, input.device={x.device}")
            return False

        # Shape check
        if result.shape != x.shape:
            print(f"SHAPE MISMATCH: result.shape={result.shape}, expected={x.shape}")
            return False

        # Numerical comparison
        # Using bfloat16, which has lower precision than float32.
        # rtol=1e-2, atol=2e-2 are appropriate for bfloat16 computations.
        rtol = 1e-2
        atol = 2e-2

        try:
            # Cast both to float32 for comparison to avoid bfloat16 comparison artifacts
            result_f32 = result.float()
            y_ref_f32 = y_ref.float()

            match = torch.allclose(result_f32, y_ref_f32, rtol=rtol, atol=atol)

            if not match:
                abs_diff = torch.abs(result_f32 - y_ref_f32)
                rel_diff = torch.abs((result_f32 - y_ref_f32) / (y_ref_f32.abs() + 1e-8))

                print(f"NUMERICAL MISMATCH:")
                print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
                print(f"  Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"  Expected (first 10): {y_ref_f32.flatten()[:10]}")
                print(f"  Got (first 10): {result_f32.flatten()[:10]}")
                print(f"  Max absolute difference: {abs_diff.max().item():.6f}")
                print(f"  Mean absolute difference: {abs_diff.mean().item():.6f}")
                print(f"  Max relative error: {rel_diff.max().item():.6f}")
                print(f"  Fraction of elements exceeding tolerance: "
                      f"{(abs_diff > atol + rtol * y_ref_f32.abs()).float().mean().item():.4f}")

                # Show worst offenders
                flat_abs = abs_diff.flatten()
                worst_idx = flat_abs.topk(5).indices
                print(f"  Worst 5 absolute differences at indices {worst_idx.tolist()}:")
                for idx in worst_idx:
                    print(f"    idx={idx.item()}: expected={y_ref_f32.flatten()[idx].item():.6f}, "
                          f"got={result_f32.flatten()[idx].item():.6f}, "
                          f"diff={flat_abs[idx].item():.6f}")
                return False

            print(f"Numerical check PASSED (rtol={rtol}, atol={atol}, dtype=bfloat16)")

        except Exception as e:
            print(f"Exception during numerical comparison: {e}")
            import traceback
            traceback.print_exc()
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