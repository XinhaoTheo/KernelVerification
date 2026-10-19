import sys, json, torch
sys.path.insert(0, "/root/cases/case_141")
import kernel

M = kernel._MULTIPLIER; S = kernel._SHIFT
# pure-python exhaustive check of the magic formula over a dense window near domain top
bad = []
for v in list(range(0, 5000)) + list(range(4294967295-20000, 4294967296)) + list(range(2**32-7*3-50, 2**32)) + list(range(2**32//7*7-100, 2**32//7*7+100)) + list(range(613566756*7-100, 613566756*7+100)):
    q = (v * M) >> S
    if q != v // 7:
        bad.append((v, q, v//7))
# GPU runtime check on boundary/biased set
vals = [0,1,6,7,8,13,14, 2**32-8,2**32-7,2**32-2,2**32-1, 2**32//7*7, 2**32//7*7+1, 2**32//7*7-1, 613566756*7, 613566756*7+6, 613566756*7+7 if 613566756*7+7 < 2**32 else 613566756*7-1, 4294967289, 4294967290, 4294967293, 4294967294, 4294967295]
vals += [max(0,(v//7*7+d)) for v in [4294967295, 2147483647, 1073741824, 3221225473] for d in (-2,-1,0,1,2)]
vals = sorted(set(x for x in vals if 0 <= x < 2**32))
x = torch.tensor(vals, dtype=torch.int64).to(torch.uint32).cuda()
out = kernel.run(x)
ref = kernel.reference(x.cpu()).cuda()
match = bool((out == ref).all())
cols = out[:,1].to(torch.int64)
col_ok = bool(((cols >= 0) & (cols <= 6)).all())
recon = out[:,0].to(torch.int64)*7 + cols
recon_ok = bool((recon == x.cpu().to(torch.int64)).all())
n_mismatch = int((out != ref).any(dim=1).sum())
print(json.dumps({"py_bad_count": len(bad), "py_bad_examples": bad[:5], "n_tested_gpu": len(vals), "gpu_match_ref": match, "col_in_range": col_ok, "recon_exact": recon_ok, "n_mismatch": n_mismatch, "multiplier": M, "shift": S}))
