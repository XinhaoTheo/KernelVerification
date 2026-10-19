import json, math, sys, importlib.util
import torch

spec = importlib.util.spec_from_file_location("k114", "/root/cases/case_114/kernel.py")
k114 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k114)
import triton

dev = "cuda"
B, QT, HQ, HK, D, PAGE = 1, 33, 2, 1, 64, 16
length, window, P = 130, 50, 16
C = math.ceil(length / PAGE)  # 9

g = torch.Generator().manual_seed(7)
q = (2*torch.rand((B, QT, HQ, D), generator=g)-1).half()
k = torch.zeros((P, PAGE, HK, D), dtype=torch.float16)
v = torch.zeros_like(k)
table = torch.arange(P, dtype=torch.int32).reshape(1, P)  # identity mapping, C=9 <= P
lengths = torch.tensor([length], dtype=torch.int32)

lo = length - QT - window + 1  # 48
# Fill allowed positions [lo, length) with finite values; poison all earlier slots with NaN.
for t in range(lo, length):
    p, s = t // PAGE, t % PAGE
    k[p, s] = (2*torch.rand((HK, D), generator=g)-1).half()
    v[p, s] = (2*torch.rand((HK, D), generator=g)-1).half()
k[:3] = float("nan"); v[:3] = float("nan")   # pages covering positions 0..47 -> NaN
for t in (q, k, v, table, lengths): t = t.to(dev)

# Direct kernel launch, bypassing validate_inputs (garbage slots violate its finite check
# but are contractually admitted by problem.txt).
b, qt, hq, d = q.shape
page, hk = k.shape[1:3]
group = hq // hk
bq = 16 // group
out = torch.empty_like(q)
k114._paged_attention[(triton.cdiv(qt, bq), hk, b)](
    q, k, v, table, lengths, out,
    QT=qt, HQ=hq, HK=hk, D=d, PAGE=page, TABLE_WIDTH=table.shape[1],
    GROUP=group, WINDOW=window, SCALE=1.0/math.sqrt(d), BQ=bq,
    num_warps=4, enable_fp_fusion=False,
)
torch.cuda.synchronize()

# Independent FP64 reference over exactly the allowed positions (ignore garbage slots).
ref = torch.empty(q.shape, dtype=torch.float64, device=dev)
for bb in range(b):
    ctx = int(lengths[bb]) - qt
    for r in range(qt):
        a = ctx + r
        end = a + 1
        begin = max(0, end - window)
        pos = torch.arange(begin, end, device=dev)
        mapped = table[bb, pos // page].long()
        kk = k[mapped, pos % page].double()
        vv = v[mapped, pos % page].double()
        for h in range(hq):
            kh = h // group
            s = (kk[:, kh] * q[bb, r, h].double()[None]).sum(-1) / math.sqrt(d)
            p_ = torch.softmax(s, dim=0)
            ref[bb, r, h] = (p_[:, None] * vv[:, kh]).sum(0)

finite = bool(torch.isfinite(out).all())
diff = (out.double() - ref).abs()
tol = 0.003 + 0.003 * ref.abs()
max_err = float(diff.max()) if finite else float("nan")
ratio = float((diff / tol).max()) if finite else float("nan")
nan_rows = torch.isnan(out).any(dim=-1)[:, :, 0].sum().item() if not finite else 0
print(json.dumps({
    "case": {"B":B,"Q":QT,"Hq":HQ,"Hkv":HK,"D":D,"S":PAGE,"length":length,"window":window,"lo":lo,
             "poisoned_positions":"[0,48) via NaN pages 0-2"},
    "metric": "output finiteness + per-element abs err vs FP64 reference (tolerance 0.003+0.003|target|)",
    "output_finite": finite, "nan_query_rows": int(nan_rows),
    "max_abs_err": max_err, "tolerance_ratio": ratio, "tol_pass": bool(finite and (diff <= tol).all().item()),
}))
