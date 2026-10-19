import torch, math, json, sys
sys.path.insert(0, '/root/cases/case_114')
import kernel

dev='cuda'
torch.manual_seed(3)
inp = kernel.make_inputs(dev, seed=5)
q,k,v,t,l = inp[:5]; w = inp[5]
snap = [x.clone() for x in (q,k,v,t,l)]
out = kernel.run(q,k,v,t,l,w)
unchanged = all(torch.equal(a,b) for a,b in zip((q,k,v,t,l), snap))
bit_unchanged = all(a.view(torch.uint8).equal(b.view(torch.uint8)) for a,b in zip((q,k,v,t,l), snap))
print(json.dumps(dict(metric="bit-for-bit input equality after run()", unchanged=bool(unchanged), bit_unchanged=bool(bit_unchanged),
                      out_shape=list(out.shape), out_dtype=str(out.dtype), out_contiguous=bool(out.is_contiguous()),
                      out_finite=bool(torch.isfinite(out).all()))))
