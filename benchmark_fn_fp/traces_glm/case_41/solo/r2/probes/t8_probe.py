import torch, math, json
torch.cuda.init()
import sys
sys.path.insert(0, '/root/cases/case_114')
from kernel import run

b, qt, hq, d, page, hk = 1, 4, 2, 64, 16, 1
pages = 16
lengths = torch.tensor([100], dtype=torch.int32)
window = 5

g = torch.Generator().manual_seed(1)
q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
k = (2*torch.rand((pages,page,hk,d), generator=g)-1).half()
v = (2*torch.rand((pages,page,hk,d), generator=g)-1).half()
table = torch.randperm(pages, generator=g)[:7].to(torch.int32).reshape(1,7)

dev='cuda'
qc,kc,vc,tc,lc = (t.to(dev) for t in (q,k,v,table,lengths))
out_clean = run(qc,kc,vc,tc,lc,window)

n = int(lc[0]); lo_allowed = n - qt - window + 1  # 92
vp = vc.clone()
poisoned = 0
for t in range(0, lo_allowed):
    pid = int(tc[0, t//page]); vp[pid, t%page] = float('nan'); poisoned += 1
# poison unreferenced pages only (not in table[:ceil(n/page)])
referenced = set(int(x) for x in tc[0, : (n+page-1)//page])
extra = 0
for pid in range(pages):
    if pid not in referenced:
        vp[pid,:] = float('nan'); extra += 1

out_pois = run(qc,kc,vp,tc,lc,window)
# kernel tile_start for qb=0: first_key=92 -> tile 1 (logical 64..127); positions 64..91 loaded but disallowed
print(json.dumps({
  "window": window, "lo_allowed": lo_allowed, "poisoned_slots": poisoned, "poisoned_pages": extra,
  "clean_finite": bool(torch.isfinite(out_clean).all()),
  "pois_finite": bool(torch.isfinite(out_pois).all()),
  "nan_count": int(torch.isnan(out_pois).sum()),
  "inf_count": int(torch.isinf(out_pois).sum()),
}))
