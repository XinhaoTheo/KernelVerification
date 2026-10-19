import json, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_141/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

cands = []
base = (1<<34)//35
for off in range(-3, 4):
    for m in range(7):
        cands.append(base + off*7 + m)
cands += [0, 6, 7, 4294967289, 4294967290, 4294967291, 4294967292, 4294967293, 4294967294, 4294967295, 4294967284, 1000000006, 2000000006]
idx = torch.tensor(sorted(set(cands)), dtype=torch.uint32, device="cuda")
out = k.run(idx).to(torch.int64).cpu()
i64 = idx.to(torch.int64).cpu()
ref = k.reference(idx.cpu()).to(torch.int64)
bad = (out != ref).any(dim=1)
col_ok = (out[:,1] >= 0) & (out[:,1] <= 6)
recon_ok = (7*out[:,0] + out[:,1]) == i64
viol = (~col_ok) | (~recon_ok) | bad
print(json.dumps({
    "num": int(i64.numel()),
    "mismatch_vs_ref_count": int(bad.sum()),
    "exactness_violation_count": int(viol.sum()),
    "all_exact": bool(int(viol.sum())==0),
    "first_bad_indices": [int(x) for x in i64[viol][:10].tolist()],
    "first_bad_out": [[int(a),int(b)] for a,b in out[viol][:10].tolist()],
    "first_bad_ref": [[int(a),int(b)] for a,b in ref[viol][:10].tolist()],
}))