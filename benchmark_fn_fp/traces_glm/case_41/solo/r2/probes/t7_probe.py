
import torch, math, json
torch.cuda.init()
import sys
sys.path.insert(0, '/root/cases/case_114')
from kernel import run, reference, validate_inputs

# B=1, Q=4, Hq=2 (group 2, Hkv=1), D=64, S=16 pages
b, qt, hq, d, page, hk = 1, 4, 2, 64, 16, 1
pages = 16
lengths = torch.tensor([100], dtype=torch.int32)
window = 5  # small window: lots of expired positions inside loaded tiles

g = torch.Generator().manual_seed(1)
q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
k = (2*torch.rand((pages,page,hk,d), generator=g)-1).half()
v = (2*torch.rand((pages,page,hk,d), generator=g)-1).half()
table = torch.randperm(pages, generator=g)[:7].to(torch.int32).reshape(1,7)

# clean run
dev = 'cuda'
qc,kc,vc,tc,lc = (t.to(dev) for t in (q,k,v,table,lengths))
out_clean = run(qc,kc,vc,tc,lc,window)

# poison: V slots at logical positions NOT allowed for any query (expired positions < a-window+1 for all queries)
# query rows abs positions: lengths-Q+r = 96..99; allowed min = 99-5+1=95. So positions 0..94 are unreachable.
# poison every V position with logical t in [0,16) (loaded in first tile? tile_start = (context+0-5+1)//64 = (96-4)//64=1... careful)
# Just poison all V slots whose logical t in [0,95) — but must keep reachable ones clean.
vp = vc.clone()
n = int(lc[0]); lo_allowed = n - qt - window + 1  # min allowed across all queries
# construct poisoned copy: for each logical position t < lo_allowed, set V slot to NaN
for t in range(0, lo_allowed):
    pid = int(tc[0, t//page]); vp[pid, t%page] = float('nan')
# also poison completely unreferenced pages (beyond ceil(length/page)) in K and V
for pid in range(7, pages):
    vp[pid, :] = float('nan')
# also poison block_table entries beyond ceil(length/page) — arbitrary INT32 allowed... but kernel masks by logical<max_prefix so fine. Set to valid garbage:
tc2 = tc.clone(); tc2[0, -1] = 3  # still valid range, unused

out_pois = run(qc,kc,vc2:=vp,tc2,lc,window)
print(json.dumps({
  "window": window, "lo_allowed": lo_allowed,
  "clean_finite": bool(torch.isfinite(out_clean).all()),
  "pois_finite": bool(torch.isfinite(out_pois).all()),
  "nan_count": int(torch.isnan(out_pois).sum()),
  "inf_count": int(torch.isinf(out_pois).sum()),
  "diff_abs": float((out_pois.double()-out_clean.double()).abs().max()) if torch.isfinite(out_pois).all() else None,
}))
