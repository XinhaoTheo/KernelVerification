
import torch, json, math, sys
sys.path.insert(0, "/root/cases/case_11")
from kernel import apply_rotary

def ref_rope(x, cos, sin, interleaved):
    # x: (seqlen, nheads, headdim), cos/sin: (seqlen, headdim//2)
    h = x.shape[-1] // 2
    c = cos[:, None, :]; s = sin[:, None, :]
    x0 = x[..., :h]; x1 = x[..., h:]
    if not interleaved:
        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c
        return torch.cat([o0, o1], dim=-1)
    else:
        x0 = x[..., 0::2].contiguous(); x1 = x[..., 1::2].contiguous()
        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c
        out = torch.empty_like(x)
        out[..., 0::2] = o0; out[..., 1::2] = o1
        return out

results = {}
torch.manual_seed(0)
for interleaved in (False, True):
    for (seqlen, nheads, headdim) in [(128, 8, 64), (3, 5, 32), (17, 2, 128), (2, 1, 6)]:
        x = torch.randn(seqlen, nheads, headdim, device="cuda", dtype=torch.float32)
        # position-dependent angles
        pos = torch.arange(seqlen, device="cuda").float()
        inv = 1.0 / (10000 ** (torch.arange(headdim//2, device="cuda").float() * 2 / headdim))
        ang = pos[:, None] * inv[None, :]
        cos = torch.cos(ang); sin = torch.sin(ang)
        out = apply_rotary(x, cos, sin, interleaved)
        ref = ref_rope(x, cos, sin, interleaved)
        err = (out - ref).abs().max().item()
        results[f"il={interleaved},{seqlen},{nheads},{headdim}"] = err

# relative-position invariant: dot(R_q_m q, R_k_n k) depends on m-n
seqlen, nheads, headdim = 64, 4, 64
pos = torch.arange(seqlen, device="cuda").float()
inv = 1.0 / (10000 ** (torch.arange(headdim//2, device="cuda").float() * 2 / headdim))
ang = pos[:, None] * inv[None, :]
cos = torch.cos(ang); sin = torch.sin(ang)
q = torch.randn(seqlen, nheads, headdim, device="cuda")
k = torch.randn(seqlen, nheads, headdim, device="cuda")
rel_ok = True
maxrel = 0.0
for interleaved in (False, True):
    qr = apply_rotary(q, cos, sin, interleaved)
    kr = apply_rotary(k, cos, sin, interleaved)
    dots = (qr[:, None] * kr[None, :]).sum(-1)  # (m, n, nheads)
    for n in range(seqlen):
        d = dots[0, n]  # m=0
        dref = dots[seqlen-1, n]  # m=S-1 -> shift; compare same relative offset
        # compute reference for relative offset n-0 via direct rotation formula
        pass
    # check invariance: dot(m,n) should equal dot(m+1,n+1)
    diff = (dots[:-1, :-1] - dots[1:, 1:]).abs().max().item()
    maxrel = max(maxrel, diff)
    rel_ok = rel_ok and diff < 1e-3

print(json.dumps({"elementwise_max_abs_err": results, "invariance_max_diff": maxrel, "invariance_ok": rel_ok}))
