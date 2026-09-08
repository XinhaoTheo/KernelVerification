
import json, math, importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_29/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'
def snap(t): return t.to(torch.float8_e4m3fn).to(torch.float32)
vals=[]
for e in range(-6,9):
    p=2.0**e
    vals.append(p)
    vals.append(float(np.nextafter(np.float32(p), np.float32(0))))
    vals.append(float(np.nextafter(np.float32(p), np.float32(np.inf))))
vals=[v for v in vals if abs(v)<=448.0]
x=torch.tensor(vals+[-v for v in vals], dtype=torch.float32, device=dev)
q=k.fp8_roundtrip(x); ref=snap(x)
offgrid=(q!=snap(q))
rel=((q-x).abs()/x.abs())
res={
 'n':int(x.numel()),
 'offgrid_count':int(offgrid.sum()),
 'mismatch_vs_ref_count':int((q!=ref).sum()),
 'max_rel_err_vs_input':float(rel.max()),
 'contract_rel_bound':0.0625,
 'count_rel_err_gt_bound':int((rel>0.0625).sum()),
}
bad=torch.nonzero(offgrid|(q!=ref)).flatten()[:12].tolist()
res['examples']=[{'x':float(x[i]),'kernel':float(q[i]),'e4m3_ref':float(ref[i]),'rel_err':float(rel[i])} for i in bad]
# dense random sweep in the pure normal range [2^-6, 448) to isolate binade selection from subnormal/overflow bugs
torch.manual_seed(1)
u=torch.empty(200000, device=dev).uniform_(math.log2(2**-6), math.log2(448.0))
xn=torch.pow(2.0,u).float()
xn=torch.cat([xn,-xn])
qn=k.fp8_roundtrip(xn); rn=snap(xn)
offn=(qn!=snap(qn))
reln=((qn-xn).abs()/xn.abs())
res['normal_range_n']=int(xn.numel())
res['normal_range_offgrid_count']=int(offn.sum())
res['normal_range_mismatch_vs_ref']=int((qn!=rn).sum())
res['normal_range_max_rel_err']=float(reln.max())
res['normal_range_count_rel_gt_bound']=int((reln>0.0625001).sum())
print(json.dumps(res))
