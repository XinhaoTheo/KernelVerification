import torch
from kernel import matmul


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"

    # Conventional test shape: K is a multiple of the K block size (32).
    a_r = torch.randn(128, 256, device=device)
    b_r = torch.randn(256, 128, device=device)
    cand_r = matmul(a_r, b_r)
    ref_r = a_r @ b_r
    divisible_pass = torch.allclose(cand_r, ref_r, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular K: the last K tile is partial.
    a_i = torch.randn(128, 250, device=device)
    b_i = torch.randn(250, 128, device=device)
    cand_i = matmul(a_i, b_i)
    ref_i = a_i @ b_i
    irregular_pass = torch.allclose(cand_i, ref_i, rtol=1e-2, atol=1e-2)
    rel = ((cand_i - ref_i).abs().max() / ref_i.abs().max()).item()

    print(f"K=256 (multiple of BLOCK_K=32): allclose={divisible_pass}")
    print(f"K=250 (partial K tile): allclose={irregular_pass}, max rel err={rel:.4f}")
    # What a conventional CI test would conclude: benchmark shapes use round K.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
