import json, math, importlib.util
import torch
import triton

spec = importlib.util.spec_from_file_location("k114", "/root/cases/case_114/kernel.py")
k114 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k114)

dev = "cuda"
B, QT, HQ, HK, D, PAGE = 1, 33, 2, 1, 64, 16
length, window, P = 130, 50, 16

g = torch.Generator().manual_seed(7)
q = (2*torch.rand((B, QT, HQ, D), generator=g)-1).half().to(dev)
k = torch.zeros((P, PAGE, HK, D), dtype=torch.float16, device=dev)
v = torch.zeros_like(k)
table = torch.arange(P, dtype=torch.int32, device=dev).reshape(1, P)
lengths = torch.tensor([length], dtype=torch.int32, device=dev)

lo = length - QT - window + 1  # 48
for t in range(lo, length):
    p, s = t // PAGE, t % PAGE
    k[p, s] = (2*torch.rand((HK, D), generator=g)-1).half()
    v[p, s] = (2*torch.rand((HK, D), generator=g)-1).half()
k[:3] = float("nan"); v[:3] = float("nan")  # positions 0..47 NaN, expired for all queries

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

ref = torch.empty(q.shape, dtype=torch.float64, device=dev)
for bb in range(b):
    ctx = int(lengths[bb]) - qt
    for r in range(qt):
        end = ctx + r + 1
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
nan_rows = int(torch.isnan(out).any(dim=-1).sum().item())
diff = (out.double() - ref).abs()
tol = 0.003 + 0.003 * ref.abs()
result = {"output_finite": finite, "nan_or_inf_elements": nan_rows,
          "max_abs_err": float(diff.max()) if finite else None,
          "tolerance_ratio": float((diff / tol).max()) if finite else None,
          "tol_pass": bool(finite and (diff <= tol).all().item())}
print(json.dumps(result))
