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
        dots = torch.einsum("mhd,nhd->mnh", qr, kr)
        # shift invariance: dot(m,n) == dot(m+1,n+1)
        shift = (dots[:-1, :-1] - dots[1:, 1:]).abs().max().item()
        # exact offset invariance: all (m, m+off) equal across m
        maxinv = 0.0
        for offv in range(1, seqlen):
            vals = dots[torch.arange(seqlen-offv), torch.arange(offv, seqlen)]  # (S-off, nheads)
            maxinv = max(maxinv, (vals - vals[0]).abs().max().item())
        # cross-check kernel vs ref elementwise again here
        res[f"{name},il={interleaved}"] = {"shift": shift, "offset_inv": maxinv}

# elementwise kernel vs ref, fp32
for interleaved in (False, True):
    out = apply_rotary(q, cos, sin, interleaved)
    ref = ref_rope(q, cos, sin, interleaved)
    res[f"elemerr,il={interleaved}"] = (out - ref).abs().max().item()

print(json.dumps(res))
