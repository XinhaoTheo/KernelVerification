import torch
from kernel import softmax


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    block = 128

    # Conventional test shape: n_cols is a multiple of the block size.
    x_round = torch.randn(8, 1024, device=device)
    cand_round = softmax(x_round, block)
    ref_round = torch.softmax(x_round, dim=1)
    divisible_pass = torch.allclose(cand_round, ref_round, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular shape: a trailing partial block exists.
    x_tail = torch.randn(8, 1000, device=device)
    cand_tail = softmax(x_tail, block)
    ref_tail = torch.softmax(x_tail, dim=1)
    irregular_pass = torch.allclose(cand_tail, ref_tail, rtol=1e-2, atol=1e-2)

    # A loose tolerance cannot expose this one even at the irregular shape:
    # softmax entries are ~1/n_cols, so atol=1e-2 swallows the whole error.
    # The contract's own invariant does expose it.
    row_sums = cand_tail.sum(dim=1)
    invariant_holds = bool(torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3))

    print(f"n_cols=1024 (multiple of block): allclose={divisible_pass}")
    print(f"n_cols=1000 (partial tail): allclose={irregular_pass} (loose tolerance still passes), "
          f"row sums {row_sums.min().item():.4f}..{row_sums.max().item():.4f}, "
          f"sums-to-one invariant holds={invariant_holds}")
    # What a conventional CI test would conclude: a round n_cols never creates the
    # trailing block, and even at n_cols=1000 the tolerance comparison passes.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and irregular_pass and not invariant_holds


if __name__ == "__main__":
    test_kernel()
