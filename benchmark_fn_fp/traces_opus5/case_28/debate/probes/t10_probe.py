
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_29/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'
def snap(t): return t.to(torch.float8_e4m3fn).to(torch.float32)
vals=[416.,448.,449.,460.,464.,500.,512.,1000.,4096.,1e4]
x=torch.tensor(vals+[-v for v in vals], dtype=torch.float32, device=dev)
q=k.fp8_roundtrip(x)
sat = snap(x.clamp(-448.,448.))
try:
    direct = snap(x)
    direct_l=[float(v) for v in direct]
except Exception as e:
    direct_l=['err:'+str(e)]
offgrid = (q != snap(q))
res={
 'x':[float(v) for v in x],
 'kernel':[float(v) for v in q],
 'saturating_ref':[float(v) for v in sat],
 'torch_direct_cast':direct_l,
 'offgrid_count':int(offgrid.sum()),
 'n':int(x.numel()),
 'mismatch_vs_saturating_ref':int((q!=sat).sum()),
 'max_kernel_abs':float(q.abs().max()),
 'e4m3_max_finite':448.0,
 'count_kernel_exceeds_448':int((q.abs()>448.).sum()),
}
# also confirm which above-448 kernel outputs are unrepresentable
res['unrep_examples']=[{'x':float(x[i]),'kernel':float(q[i]),'snap(kernel)':float(snap(q)[i])} for i in torch.nonzero(offgrid).flatten()[:10].tolist()]
print(json.dumps(res))
