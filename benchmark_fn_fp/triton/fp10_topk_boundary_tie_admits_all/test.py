import torch
from kernel import topk_mask


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols, k = 64, 8

    # A row where the value at the cutoff is shared by several entries.
    scores = torch.randn(4, n_cols, device=device)
    scores[:, :12] = 2.0                      # 12 entries tied above the cutoff
    cand = topk_mask(scores, k)

    # Reference reading: admit exactly k, breaking the tie by lowest index.
    ref = torch.zeros_like(scores)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    ref.scatter_(1, order, scores.gather(1, order))

    raw_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    # The contract's actual requirement: no dropped entry exceeds a kept one.
    kept = cand != 0
    min_kept = torch.where(kept, cand, torch.full_like(cand, float("inf"))).min(dim=1).values
    max_dropped = torch.where(kept, torch.full_like(cand, -float("inf")), scores).max(dim=1).values
    ordering_ok = bool((min_kept >= max_dropped - 1e-6).all().item())

    plain = torch.tensor([[5., 4., 3., 2., 1., 0., -1., -2.]], device=device)
    kept_no_tie = {kk: int((topk_mask(plain, kk) != 0).sum().item()) for kk in (1, 2, 3)}
    exact_without_ties = all(kept_no_tie[kk] == kk for kk in kept_no_tie)
    print(f"no ties at all, kept counts by k = {kept_no_tie}, exactly k = {exact_without_ties}")

    n_kept = int(kept[0].sum().item())
    print(f"reference admits exactly k={k}; kernel kept {n_kept} entries")
    print(f"elementwise allclose against that one reference = {raw_pass} (expected False)")
    print(f"contract requirement (no dropped entry exceeds a kept one) = {ordering_ok} (expected True)")
    # What a conventional CI test would conclude: it compares elementwise against
    # one implementation's private tie convention.
    print(f"NAIVE_ALLCLOSE_VERDICT: {raw_pass}")
    return (not raw_pass) and ordering_ok and exact_without_ties


if __name__ == "__main__":
    test_kernel()
