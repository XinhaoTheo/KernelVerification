import torch
from kernel import chunked_cumsum


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    chunk = 64

    # Conventional test shape: seqlen is a multiple of the chunk size.
    x_r = torch.randn(4, 512, device=device)
    cand_r = chunked_cumsum(x_r, chunk)
    ref_r = x_r.cumsum(dim=1)
    divisible_pass = torch.allclose(cand_r, ref_r, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular length: a shorter final chunk exists.
    x_i = torch.randn(4, 500, device=device)
    cand_i = chunked_cumsum(x_i, chunk)
    ref_i = x_i.cumsum(dim=1)
    irregular_pass = torch.allclose(cand_i, ref_i, rtol=1e-2, atol=1e-2)
    tail_err = (cand_i[:, 448:] - ref_i[:, 448:]).abs().max().item()

    print(f"seqlen=512 (multiple of chunk): allclose={divisible_pass}")
    print(f"seqlen=500 (short final chunk): allclose={irregular_pass}, tail max abs err={tail_err:.4f}")
    # What a conventional CI test would conclude: benchmark sequence lengths are round.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
