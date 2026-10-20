import torch, numpy as np, json, sys, glob
sys.path.insert(0, glob.glob('/root/cases/case_102')[0] if glob.glob('/root/cases/case_102') else '/root/cases/case_102')
import kernel as K

dev = 'cuda'
# Exhaustive sweep of ALL finite FP16 bit patterns in chunks <=131072
bits = np.arange(0x0000, 0x10000, dtype=np.uint16)
vals = torch.from_numpy(bits.view(np.float16).copy())
finite = torch.isfinite(vals)
fv = vals[finite].to(dev)
print("total patterns", bits.size, "finite", int(finite.sum().item()))

n = fv.numel()
print("n =", n)  # should be ~64512-65504 depending
chunks = [fv[i:i+131072] for i in range(0, n, 131072)]
out_parts = [K.run(c) for c in chunks]
out = torch.cat(out_parts)
ref = K.reference(fv).to(dev)  # FP64
rel = ((out.to(torch.float64) - ref).abs() / ref.abs())
max_rel = rel.max().item()
bad = (rel > 1e-6)
print("max_rel", max_rel, "num_bad", int(bad.sum().item()), "tol", 1e-6)
idx = torch.argsort(rel, descending=True)[:5]
print("worst x:", [float(fv[i]) for i in idx.tolist()])
print("worst rel:", [float(rel[i]) for i in idx.tolist()])
print(json.dumps({"max_rel": max_rel, "tolerance": 1e-6, "num_bad": int(bad.sum().item()), "n_inputs": n,
                  "worst_x": [float(fv[i]) for i in idx.tolist()], "worst_rel": [float(rel[i]) for i in idx.tolist()]}))