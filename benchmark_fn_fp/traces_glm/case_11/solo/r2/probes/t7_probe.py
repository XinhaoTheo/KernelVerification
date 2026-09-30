import torch, json, sys
sys.path.insert(0, "/root/cases/case_11")
from kernel import apply_rotary

def ref_rope(x, cos, sin, interleaved):
    h = x.shape[-1] // 2
    c = cos[:, None, :]; s = sin[:, None, :]
    if not interleaved:
        x0 = x[..., :h]; x1 = x[..., h:]
        return torch.cat([x0*c - x1*s, x0*s + x1*c], dim=-1)
    x0 = x[..., 0::2]; x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0*c - x1*s; out[..., 1::2] = x0*s + x1*c
    return out

seqlen, nheads, headdim = 64, 4, 64
pos = torch.arange(seqlen, device="cuda").float()
inv = 1.0/(10000**(torch.arange(headdim//2, device="cuda").float()*2/headdim))
ang = pos[:, None]*inv[None, :]
cos = torch.cos(ang); sin = torch.sin(ang)
torch.manual_seed(0)
q = torch.randn(seqlen, nheads, headdim, device="cuda")
k = torch.randn(seqlen, nheads, headdim, device="cuda")

res = {}
for name, fn in [("ref", ref_rope), ("kernel", apply_rotary)]:
    for interleaved in (False, True):
        qr = fn(q, cos, sin, interleaved)
        kr = fn(k, cos, sin, interleaved)
        dots = (qr[:, None] * kr[None, :]).sum(-1)
        diff = (dots[:-1, :-1] - dots[1:, 1:]).abs().max().item()
        res[f"{name},il={interleaved}"] = diff
        # stronger invariant check: dot(m,n) depends only on m-n
        d0 = dots[0, :-1]  # relative offset 1..S-1
        dcols = torch.stack([dots[m, m+1:seqlen] for m in range(seqlen-1)])
        off = dcols - d0[None, :len(dcols[0]) if False else None]
        # simpler: compare all (m,n) with same offset
        maxinv = 0.0
        for offv in range(1, seqlen):
            vals = torch.stack([dots[m, m+offv] for m in range(seqlen-offv)])
            maxinv = max(maxinv, (vals - vals[0]).abs().max().item())
        res[f"{name},il={interleaved},shift_inv"] = maxinv
print(json.dumps(res))
