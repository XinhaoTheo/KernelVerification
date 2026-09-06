import torch
from kernel import group_quant_dequant


def reference(x, group_size):
    out = torch.zeros_like(x)
    n_cols = x.shape[1]
    for start in range(0, n_cols, group_size):
        stop = min(start + group_size, n_cols)
        chunk = x[:, start:stop]
        scale = chunk.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.round(chunk / scale).clamp(-128, 127)
        out[:, start:stop] = q * scale
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    group = 64

    # Conventional test shape: n_cols is a multiple of group_size.
    x_r = torch.randn(8, 512, device=device)
    cand_r = group_quant_dequant(x_r, group)
    ref_r = reference(x_r, group)
    divisible_pass = torch.allclose(cand_r, ref_r, rtol=1e-2, atol=1e-2)

    # Contract-permitted irregular shape: a shorter final group exists.
    x_i = torch.randn(8, 500, device=device)
    cand_i = group_quant_dequant(x_i, group)
    ref_i = reference(x_i, group)
    irregular_pass = torch.allclose(cand_i, ref_i, rtol=1e-2, atol=1e-2)
    tail_err = (cand_i[:, 448:] - ref_i[:, 448:]).abs().max().item()

    print(f"n_cols=512 (multiple of group): allclose={divisible_pass}")
    print(f"n_cols=500 (short final group): allclose={irregular_pass}, tail max abs err={tail_err:.4f}")
    # What a conventional CI test would conclude: hidden sizes are round.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
