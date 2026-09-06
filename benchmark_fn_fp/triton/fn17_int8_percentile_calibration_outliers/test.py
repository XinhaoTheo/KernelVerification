import torch
from kernel import quant_dequant


def rel_error(y, x):
    return ((y - x).norm(dim=1) / x.norm(dim=1)).max().item()


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols = 512

    # Conventional test data: IID Gaussian, no outlier channels.
    x_iid = torch.randn(8, n_cols, device=device)
    y_iid = quant_dequant(x_iid)
    iid_rel = rel_error(y_iid, x_iid)
    iid_pass = iid_rel <= 0.05

    # Realistic activations: a few channels an order of magnitude larger, the
    # outlier structure documented by LLM.int8() and SmoothQuant.
    x_out = torch.randn(8, n_cols, device=device)
    x_out[:, 7] *= 40.0
    x_out[:, 200] *= 25.0
    y_out = quant_dequant(x_out)
    out_rel = rel_error(y_out, x_out)
    outlier_pass = out_rel <= 0.05
    worst = (y_out - x_out).abs().max().item()

    print(f"IID Gaussian rows: relative reconstruction error = {iid_rel:.4f}, within 5% = {iid_pass}")
    print(f"rows with outlier channels: relative reconstruction error = {out_rel:.4f}, "
          f"within 5% = {outlier_pass}, worst single-entry error = {worst:.2f}")
    # What a conventional CI test would conclude: it calibrates on randn, whose
    # bulk and maximum are close together.
    print(f"NAIVE_ALLCLOSE_VERDICT: {iid_pass}")
    return iid_pass and not outlier_pass


if __name__ == "__main__":
    test_kernel()
