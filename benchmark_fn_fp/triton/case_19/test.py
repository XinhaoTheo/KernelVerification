import torch
from kernel import cross_entropy_with_predictions


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n_rows, n_cols = 4, 262144          # Gemma-3 scale vocabulary: several blocks per row
    embed_dim = 16

    # --- conventional test: random logits, ties essentially never occur -------
    logits = torch.randn(n_rows, n_cols, device=device, dtype=torch.float32)
    target = torch.randint(0, n_cols, (n_rows,), device=device, dtype=torch.int64)
    _, pred_random = cross_entropy_with_predictions(logits.clone(), target)
    ref_random = logits.argmax(dim=1).to(torch.int32)
    random_pass = torch.equal(pred_random, ref_random)

    # --- contract case: an exact tie split across two blocks ------------------
    tied = torch.full((n_rows, n_cols), -10.0, device=device, dtype=torch.float32)
    low_idx, high_idx = 1234, 200000     # block 0 and block 3 at BLOCK_SIZE=65536
    tied[:, low_idx] = 5.0
    tied[:, high_idx] = 5.0
    _, pred_tie = cross_entropy_with_predictions(tied.clone(), target)

    picked = pred_tie[0].item()
    lowest_index_wins = bool((pred_tie == low_idx).all().item())

    # Downstream continuous output: the embedding row the decoder would gather.
    embedding = torch.randn(n_cols, embed_dim, device=device, dtype=torch.float32)
    got = embedding[pred_tie.to(torch.int64)]
    want = embedding[torch.full_like(pred_tie, low_idx, dtype=torch.int64)]
    downstream_pass = torch.allclose(got, want, rtol=1e-2, atol=1e-2)
    downstream_diff = (got - want).abs().max().item()

    print(f"random logits (no ties): argmax matches torch = {random_pass}")
    print(f"tie between idx {low_idx} (block 0) and idx {high_idx} (block 3): kernel picked {picked}, "
          f"lowest-index-wins = {lowest_index_wins}")
    print(f"downstream embedding allclose = {downstream_pass}, max abs diff = {downstream_diff:.4f}")
    # What a conventional CI test would conclude: random logits never tie.
    print(f"NAIVE_ALLCLOSE_VERDICT: {random_pass}")
    return random_pass and not lowest_index_wins and not downstream_pass


if __name__ == "__main__":
    test_kernel()
