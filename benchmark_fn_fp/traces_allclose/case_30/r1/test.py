import torch
from kernel import block_sparse_attention


def reference(q, k, v, cu_q, cu_k, block_mask, qbs, kbs, scale):
    """Dense per-sequence attention restricted to the blocks the mask selects."""
    out = torch.zeros_like(q)
    nh_q, nh_k = q.shape[1], k.shape[1]
    ratio = nh_q // nh_k
    for b in range(len(cu_q) - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        n_qb_before = sum((int(cu_q[i + 1]) - int(cu_q[i]) + qbs - 1) // qbs for i in range(b))
        n_kb_before = sum((int(cu_k[i + 1]) - int(cu_k[i]) + kbs - 1) // kbs for i in range(b))
        for h in range(nh_q):
            hk = h // ratio
            qb = q[qs:qe, h].float()
            kb = k[ks:ke, hk].float()
            vb = v[ks:ke, hk].float()
            scores = (qb @ kb.T) * scale
            allow = torch.zeros_like(scores, dtype=torch.bool)
            for i in range((qe - qs + qbs - 1) // qbs):
                for j in range((ke - ks + kbs - 1) // kbs):
                    if block_mask[hk, n_qb_before + i, n_kb_before + j]:
                        allow[i * qbs:(i + 1) * qbs, j * kbs:(j + 1) * kbs] = True
            scores = scores.masked_fill(~allow, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            p = torch.nan_to_num(p, nan=0.0)
            out[qs:qe, h] = (p @ vb).to(out.dtype)
    return out


def run(lengths, qbs=64, kbs=64, nh_q=4, nh_k=2, headdim=32, device="cuda"):
    torch.manual_seed(0)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=device, dtype=torch.int32)
    total = int(cu[-1])
    q = torch.randn(total, nh_q, headdim, device=device, dtype=torch.float32)
    k = torch.randn(total, nh_k, headdim, device=device, dtype=torch.float32)
    v = torch.randn(total, nh_k, headdim, device=device, dtype=torch.float32)
    n_qb = sum((L + qbs - 1) // qbs for L in lengths)
    n_kb = sum((L + kbs - 1) // kbs for L in lengths)
    mask = torch.ones(nh_k, n_qb, n_kb, dtype=torch.bool, device=device)
    scale = headdim ** -0.5
    cand = block_sparse_attention(q, k, v, cu, cu, mask, qbs, kbs, False, scale)
    ref = reference(q, k, v, cu, cu, mask, qbs, kbs, scale)
    return cand, ref


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"

    # Conventional test: a uniform batch whose sequence length is a multiple of
    # the block size. This is what benchmark harnesses generate.
    cand_u, ref_u = run([256, 256])
    uniform_pass = torch.allclose(cand_u, ref_u, rtol=1e-2, atol=1e-2)

    # Contract-permitted packed batch: lengths differ and neither is a multiple
    # of the block size, while their total happens to be one.
    cand_v, ref_v = run([100, 156])
    varlen_pass = torch.allclose(cand_v, ref_v, rtol=1e-2, atol=1e-2)
    diff = (cand_v - ref_v).abs().max().item()

    # A packed batch whose total is also not a multiple, for contrast.
    cand_w, ref_w = run([100, 150])
    other_pass = torch.allclose(cand_w, ref_w, rtol=1e-2, atol=1e-2)

    print(f"uniform batch [256, 256] (each a multiple of 64): allclose={uniform_pass}")
    print(f"packed batch [100, 156] (total 256 is a multiple of 64, neither length is): "
          f"allclose={varlen_pass}, max abs diff={diff:.4f}")
    print(f"packed batch [100, 150] (total 250 is not a multiple): allclose={other_pass}")
    # What a conventional CI test would conclude: it uses a uniform batch of
    # round sequence lengths.
    print(f"NAIVE_ALLCLOSE_VERDICT: {uniform_pass}")
    return uniform_pass and not varlen_pass


if __name__ == "__main__":
    test_kernel()
