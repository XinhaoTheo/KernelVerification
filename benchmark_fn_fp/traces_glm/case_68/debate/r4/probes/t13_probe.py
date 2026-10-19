import sys, json, torch
sys.path.insert(0, "/root/cases/case_141")
import kernel
M = kernel._MULTIPLIER; S = kernel._SHIFT
bad = []
for v in list(range(0, 20000)) + list(range(4294967295-30000, 4294967296)) + list(range(2**32//7*7-200, 2**32//7*7+200)) + list(range(613566756*7-200, 613566756*7+200)):
    if ((v * M) >> S) != v // 7:
        bad.append(v)
vals = [0,1,6,7,8,13,14, 2**32-8,2**32-7,2**32-2,2**32-1, 4294967289,4294967290,4294967293,4294967294,4294967295, 2**32//7*7-1, 613566756*7, 613566756*7+6]
x64 = torch.tensor(sorted(set(vals)), dtype=torch.int64)
x = x64.to(torch.uint32).cuda()
out = kernel.run(x)
ref = kernel.reference(x.cpu()).cuda()
mism = (out != ref).any(dim=1)
cols = out[:,1].to(torch.int64).cpu()
xc = x.cpu().to(torch.int64)
recon_ok = bool(((out[:,0].to(torch.int64).cpu()*7 + cols) == xc).all())
ex = [[int(xc[i]), int(out[i,0]), int(out[i,1])] for i in mism.nonzero().flatten()[:6]]
print(json.dumps({"py_bad_count": len(bad), "e_M": 5*M, "two_pow_shift": 1<<S, "gm_criterion_holds": 5*M <= (1<<S), "n_gpu_tested": int(x.numel()), "gpu_mismatch": int(mism.sum()), "gpu_examples": ex, "recon_exact_gpu": recon_ok, "col_range_ok_gpu": bool(((cols>=0)&(cols<=6)).all())}))