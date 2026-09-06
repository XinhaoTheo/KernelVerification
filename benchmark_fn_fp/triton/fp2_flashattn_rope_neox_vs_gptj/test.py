import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import apply_rotary  # real flash-attention rotary kernel, both conventions


def make_cos_sin(seq_len, half, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def to_interleaved(x, half):
    out = torch.empty_like(x)
    out[..., 0::2] = x[..., :half]
    out[..., 1::2] = x[..., half:]
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)
    seq_len, nheads, headdim = 4, 2, 8
    half = headdim // 2

    q = torch.randn(seq_len, nheads, headdim, device=device)
    k = torch.randn(seq_len, nheads, headdim, device=device)
    cos, sin = make_cos_sin(seq_len, half, device=device)

    q_ref = apply_rotary(q, cos, sin, interleaved=False)
    k_ref = apply_rotary(k, cos, sin, interleaved=False)

    q_cand = apply_rotary(to_interleaved(q, half), cos, sin, interleaved=True)
    k_cand = apply_rotary(to_interleaved(k, half), cos, sin, interleaved=True)

    raw_pass = torch.allclose(q_cand, q_ref, rtol=1e-2, atol=1e-2)

    scores_ref = torch.einsum("shd,thd->hst", q_ref, k_ref)
    scores_cand = torch.einsum("shd,thd->hst", q_cand, k_cand)
    downstream_pass = torch.allclose(scores_cand, scores_ref, rtol=1e-2, atol=1e-2)

    print(f"real flash-attention rope kernel (GPU): raw rotated-tensor allclose = {raw_pass} (expected False)")
    print(f"downstream attention-score allclose = {downstream_pass} (expected True)")
    print("FP DEMONSTRATED" if (not raw_pass) and downstream_pass else "tune constants")
    # What a conventional CI test would conclude: 直接比原始旋转张量，未对齐布局
    print(f"NAIVE_ALLCLOSE_VERDICT: {raw_pass}")
    return (not raw_pass) and downstream_pass


if __name__ == "__main__":
    test_kernel()
