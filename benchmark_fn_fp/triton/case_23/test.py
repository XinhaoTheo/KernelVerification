import torch
from kernel import cosine_similarity


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_cols = 512

    # Rows whose vectors are near-orthogonal, so the true cosine is ~0.
    a = torch.randn(64, n_cols, device=device)
    r = torch.randn_like(a)
    # Project r onto the orthogonal complement of a: the true cosine is ~0.
    b = r - ((r * a).sum(1, keepdim=True) / (a * a).sum(1, keepdim=True)) * a
    cand = cosine_similarity(a, b)
    ref = torch.nn.functional.cosine_similarity(a, b, dim=1)

    abs_diff = (cand - ref).abs().max().item()
    ref_mag = ref.abs().max().item()
    rel = ((cand - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()

    relative_only_pass = rel <= 1e-2
    combined_pass = torch.allclose(cand, ref, rtol=1e-2, atol=1e-2)

    print(f"true cosine magnitude ~ {ref_mag:.3e}, max abs diff = {abs_diff:.3e}")
    print(f"relative-error-only check = {relative_only_pass} (expected False), max rel = {rel:.3f}")
    print(f"torch.allclose (atol+rtol combined) = {combined_pass} (expected True)")
    # What a conventional CI test would conclude: it uses a relative-error
    # metric, which is ill-conditioned when the true value is ~0.
    print(f"NAIVE_ALLCLOSE_VERDICT: {relative_only_pass}")
    return (not relative_only_pass) and combined_pass


if __name__ == "__main__":
    test_kernel()
