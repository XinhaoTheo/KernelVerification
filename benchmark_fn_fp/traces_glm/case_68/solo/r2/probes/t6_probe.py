import json, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_141/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# Candidate failing indices: near threshold 2^34/35 ~ 4.912e8, mod 7 == 6
cands = []
base = (1<<34)//35
for off in range(-3, 4):
    for m in range(7):
        cands.append(base + off*7 + m)
cands += [0, 6, 7, 4294967289, 4294967290, 4294967291, 4294967292, 4294967293, 4294967294, 4294967295, 4294967291-7, 1000000000+6, 2000000000+6]
idx = torch.tensor(sorted(set(cands)), dtype=torch.uint32, device="cuda")
out = k.run(idx)
ref = k.reference(idx.cpu()).to("cuda")
bad = (out != ref).any(dim=1)
viol = ((out[:,1] > 6) | ((7*out[:,0].to(torch.int64) + out[:,1].to(torch.int64)).to(torch.uint32) != idx))
result = {
    "num": idx.numel(),
    "mismatch_vs_ref_count": int(bad.sum()),
    "col_range_violation_count": int(viol.sum()),
    "first_bad_indices": [int(x) for x in idx[viol][:10].tolist()],
    "first_bad_out": [[int(a),int(b)] for a,b in out[viol][:10].tolist()],
    "first_bad_ref": [[int(a),int(b)] for a,b in ref[viol][:10].tolist()],
}
print(json.dumps(result))
