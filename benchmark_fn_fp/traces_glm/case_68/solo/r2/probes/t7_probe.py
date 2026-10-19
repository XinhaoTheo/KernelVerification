import json, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_141/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

cands = []
base = (1<<34)//35
for off in range(-3, 4):
    for m in range(7):
        cands.append(base + off*7 + m)
cands += [0, 6, 7, 4294967289, 4294967290, 4294967291, 4294967292, 4294967293, 4294967294, 4294967295, 4294967291-7, 1000000000+6, 2000000000+6]
idx = torch.tensor(sorted(set(cands)), dtype=torch.uint32, device="cuda")
out = k.run(idx)
o64 = out.to(torch.int64); i64 = idx.to(torch.int64)
ref = k.reference(idx.cpu()).to(torch.int64).to("cuda")
bad = (o64 != ref).any(dim=1)
col_ok = (o64[:,1] >= 0) & (o64[:,1] <= 6)
recon_ok = (7*o64[:,0] + o64[:,1]) == i64
viol = (~col_ok) | (~recon_ok)
print(json.dumps({
    "num": idx.numel(),
    "mismatch_vs_ref_count": int(bad.sum()),
    "exactness_violation_count": int(viol.sum()),
    "first_bad_indices": [int(x) for x in idx[viol][:10].tolist()],
    "first_bad_out": [[int(a),int(b)] for a,b in out[viol][:10].tolist()],
    "first_bad_ref": [[int(a),int(b)] for a,b in ref[viol][:10].tolist()],
    "all_exact": bool((viol).sum()==0),
}))