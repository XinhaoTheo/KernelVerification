import torch
from kernel import masked_softmax


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_rows, n_cols = 8, 64

    x = torch.randn(n_rows, n_cols, device=device)
    # Conventional test: a random mask, every row keeps something.
    mask = (torch.rand(n_rows, n_cols, device=device) > 0.3).to(torch.int32)
    mask[:, 0] = 1
    cand = masked_softmax(x, mask)
    ref = torch.softmax(x.masked_fill(mask == 0, float("-inf")), dim=1)
    common_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    # Contract-permitted rare input: one row keeps nothing.
    mask2 = mask.clone()
    mask2[3, :] = 0
    cand2 = masked_softmax(x, mask2)
    row = cand2[3]
    all_finite = bool(torch.isfinite(cand2).all().item())
    zero_row = bool((row.abs() < 1e-6).all().item())

    # Downstream: the row is used to average a value matrix.
    v = torch.randn(n_cols, 16, device=device)
    out = cand2 @ v
    downstream_finite = bool(torch.isfinite(out).all().item())

    print(f"random mask, every row keeps something: allclose={common_pass}")
    print(f"fully masked row: all finite={all_finite}, all-zero as specified={zero_row}, "
          f"row[0]={row[0].item()}")
    print(f"downstream value average finite={downstream_finite}")
    # What a conventional CI test would conclude: its random mask never empties a row.
    print(f"NAIVE_ALLCLOSE_VERDICT: {common_pass}")
    return common_pass and not (all_finite and zero_row)


if __name__ == "__main__":
    test_kernel()
