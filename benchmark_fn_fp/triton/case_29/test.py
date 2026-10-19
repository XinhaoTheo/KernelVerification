import torch
from kernel import route_top1


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_tokens, n_experts, dim = 64, 8, 32

    # Conventional test: float32 random logits, exact ties essentially never occur.
    logits = torch.randn(n_tokens, n_experts, device=device)
    cand = route_top1(logits)
    ref = logits.argmax(dim=1).to(torch.int32)
    random_pass = torch.equal(cand, ref)

    # Reduced-precision logits: distinct values collapse onto the same bf16 level.
    tied = torch.randn(n_tokens, n_experts, device=device).to(torch.bfloat16).float()
    tied[:, 2] = tied[:, 5] = tied.max(dim=1).values + 1.0
    cand_t = route_top1(tied)
    lowest_wins = bool((cand_t == 2).all().item())

    # Downstream: independently trained experts, so the choice moves the output.
    experts = torch.randn(n_experts, dim, device=device)
    got = experts[cand_t.to(torch.int64)]
    want = experts[torch.full_like(cand_t, 2, dtype=torch.int64)]
    downstream_pass = torch.allclose(got, want, rtol=1e-2, atol=1e-2)

    print(f"float32 random logits (no ties): matches torch.argmax = {random_pass}")
    print(f"tie between expert 2 and expert 5: kernel picked {cand_t[0].item()}, "
          f"lowest-index-wins = {lowest_wins}")
    print(f"downstream expert output allclose = {downstream_pass}")
    # What a conventional CI test would conclude: float32 logits do not tie.
    print(f"NAIVE_ALLCLOSE_VERDICT: {random_pass}")
    return random_pass and not lowest_wins and not downstream_pass


if __name__ == "__main__":
    test_kernel()
