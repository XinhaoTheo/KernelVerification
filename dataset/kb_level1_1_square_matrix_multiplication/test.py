import torch
import torch.nn as nn

# Summary: Test for a Triton kernel implementing square matrix multiplication C = A * B
# where A and B are square matrices of shape (N, N) with N = 2048 * 2 = 4096.
# The kernel should produce results matching torch.matmul(A, B).

def test_kernel():
    """Test the kernel implementation for square matrix multiplication."""
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
        N = 2048 * 2  # N = 4096

        print(f"Testing square matrix multiplication with N={N}")
        print(f"Matrix A shape: ({N}, {N}), Matrix B shape: ({N}, {N})")

        # Use BF16 instead of FP32 (as per requirements)
        # Using torch.rand as specified in get_inputs()
        A = torch.rand(N, N, dtype=torch.bfloat16, device=device)
        B = torch.rand(N, N, dtype=torch.bfloat16, device=device)

        print(f"A dtype: {A.dtype}, device: {A.device}")
        print(f"B dtype: {B.dtype}, device: {B.device}")
        print(f"A sample values (first 5): {A.flatten()[:5]}")
        print(f"B sample values (first 5): {B.flatten()[:5]}")

        # Compute reference output using PyTorch's matmul
        # Reference model as described in the problem
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                return torch.matmul(A, B)

        ref_model = Model().to(device)
        with torch.no_grad():
            y_ref = ref_model(A, B)

        print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
        print(f"Reference output sample (first 5): {y_ref.flatten()[:5]}")

        # Call kernel_function as a normal Python function
        # Pass A and B directly (NOT the model or precomputed reference)
        result = kernel_function(A, B)

        print(f"Kernel output shape: {result.shape}, dtype: {result.dtype}")
        print(f"Kernel output sample (first 5): {result.flatten()[:5]}")

        # Device check
        if isinstance(result, torch.Tensor):
            if result.device.type != A.device.type:
                print(f"Device mismatch: result is on {result.device}, expected {A.device}")
                return False
        else:
            print(f"Expected a torch.Tensor, got {type(result)}")
            return False

        # Shape check
        expected_shape = (N, N)
        if result.shape != torch.Size(expected_shape):
            print(f"Shape mismatch: expected {expected_shape}, got {result.shape}")
            return False

        # Numerical comparison
        # Using looser tolerances for BF16 due to lower precision (16-bit)
        # Additionally, N=4096 means we accumulate over 4096 elements per output,
        # which can amplify floating point errors significantly.
        # For BF16 with large accumulation dimension (4096), we use rtol=1e-1, atol=1e-1
        rtol = 1e-1
        atol = 1e-1

        # Cast result to match reference dtype for comparison if needed
        result_for_compare = result
        if result.dtype != y_ref.dtype:
            print(f"Warning: dtype mismatch between result ({result.dtype}) and reference ({y_ref.dtype}). Casting result for comparison.")
            result_for_compare = result.to(y_ref.dtype)

        try:
            if not torch.allclose(result_for_compare, y_ref, rtol=rtol, atol=atol):
                print(f"NUMERICAL MISMATCH:")
                print(f"Input A shape: {A.shape}, dtype: {A.dtype}")
                print(f"Input B shape: {B.shape}, dtype: {B.dtype}")
                print(f"Expected shape: {y_ref.shape}, dtype: {y_ref.dtype}")
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                print(f"Expected (first 10): {y_ref.flatten()[:10]}")
                print(f"Got (first 10): {result_for_compare.flatten()[:10]}")
                abs_diff = torch.abs(result_for_compare - y_ref)
                max_abs_diff = torch.max(abs_diff).item()
                mean_abs_diff = torch.mean(abs_diff).item()
                print(f"Max absolute difference: {max_abs_diff}")
                print(f"Mean absolute difference: {mean_abs_diff}")
                # Avoid division by zero in relative error
                rel_err = torch.max(torch.abs(result_for_compare - y_ref) / (torch.abs(y_ref) + 1e-8)).item()
                print(f"Max relative error: {rel_err}")
                # Show where the worst mismatches are
                flat_diff = abs_diff.flatten()
                top_k_indices = torch.topk(flat_diff, min(5, flat_diff.numel())).indices
                print(f"Top mismatch indices: {top_k_indices.tolist()}")
                print(f"Expected at those indices: {y_ref.flatten()[top_k_indices]}")
                print(f"Got at those indices: {result_for_compare.flatten()[top_k_indices]}")
                return False
            else:
                print(f"Numerical check PASSED (rtol={rtol}, atol={atol})")
        except Exception as cmp_e:
            print(f"Error during numerical comparison: {cmp_e}")
            return False

        print("All checks passed!")
        return True

    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        else:
            import traceback
            print(f"Test failed: {type(e).__name__}: {e}")
            traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)