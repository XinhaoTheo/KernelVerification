import torch
from kernel import triton_argsort


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    N, W = 8, 64

    # Conventional test: keys drawn from a wide range, so ties are vanishingly rare.
    keys = torch.randint(0, 2**40, (N, W), device=device, dtype=torch.int64)
    _, perm = triton_argsort(keys.clone())
    ref_perm = torch.argsort(keys, dim=1, stable=True)
    distinct_pass = torch.equal(perm, ref_perm.to(perm.dtype))

    # Contract case: quantized keys, so a row holds many duplicates.
    tied = torch.randint(0, 4, (N, W), device=device, dtype=torch.int64)
    sorted_t, perm_t = triton_argsort(tied.clone())
    ref_t = torch.argsort(tied, dim=1, stable=True)
    sorted_ok = bool((sorted_t[:, 1:] >= sorted_t[:, :-1]).all().item())
    stable = torch.equal(perm_t, ref_t.to(perm_t.dtype))

    # Downstream: the permutation gathers memory slots whose contents differ.
    slots = torch.randn(W, 16, device=device)
    got = slots[perm_t.reshape(-1).clamp_(0, W - 1)]
    want = slots[ref_t.reshape(-1)]
    downstream_pass = torch.allclose(got, want, rtol=1e-2, atol=1e-2)
    diff = (got - want).abs().max().item()

    print(f"wide-range keys (no ties): permutation matches stable argsort = {distinct_pass}")
    print(f"quantized keys (many ties): keys still sorted = {sorted_ok}, stable = {stable}")
    print(f"downstream gathered slots allclose = {downstream_pass}, max abs diff = {diff:.4f}")
    # What a conventional CI test would conclude: random wide-range keys do not tie.
    print(f"NAIVE_ALLCLOSE_VERDICT: {distinct_pass}")
    return distinct_pass and sorted_ok and not stable and not downstream_pass


if __name__ == "__main__":
    test_kernel()
