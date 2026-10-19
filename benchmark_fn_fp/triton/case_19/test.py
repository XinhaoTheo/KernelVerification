import torch
from kernel import layer_norm


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols, eps = 512, 1e-5
    w = torch.ones(n_cols, device=device)
    b = torch.zeros(n_cols, device=device)

    x_normal = torch.randn(8, n_cols, device=device)
    cand_n = layer_norm(x_normal, w, b, eps)
    ref_n = torch.nn.functional.layer_norm(x_normal, (n_cols,), w, b, eps)
    normal_pass = torch.allclose(cand_n, ref_n, rtol=1e-2, atol=1e-2)

    # A row whose variance is far below eps, but whose deviations are still
    # representable, so the normalized output is not identically zero.
    x_flat = torch.full((1, n_cols), 1e-4, device=device)
    x_flat[:, 1::2] = -1e-4
    cand_f = layer_norm(x_flat, w, b, eps)
    ref_f = torch.nn.functional.layer_norm(x_flat, (n_cols,), w, b, eps)
    flat_pass = torch.allclose(cand_f, ref_f, rtol=1e-2, atol=1e-2)

    print(f"normal rows: allclose={normal_pass}")
    print(f"near-constant row: allclose={flat_pass}, ref max={ref_f.abs().max().item():.3e}, cand max={cand_f.abs().max().item():.3e}")
    # What a conventional CI test would conclude: Gaussian rows never have
    # near-zero variance.
    print(f"NAIVE_ALLCLOSE_VERDICT: {normal_pass}")
    return normal_pass and not flat_pass


if __name__ == "__main__":
    test_kernel()
