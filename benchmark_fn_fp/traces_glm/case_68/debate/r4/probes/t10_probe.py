import sys, json, torch
sys.path.insert(0, "/root/cases/case_141")
import kernel
g = torch.Generator().manual_seed(1)
# random large values (top 1% and top 90% of domain) where 32-bit truncation would show
hi = (torch.rand(20000, generator=g) * 2**32).to(torch.int64)
vals = torch.cat([hi[hi > 2**31][:5000], hi[:5000], torch.tensor([2**32-1, 2**32-2, 4294967289, 2**31, 2**31+7, 2**31-1], dtype=torch.int64)])
vals = torch.unique(vals).to(torch.uint32).cuda()
out = kernel.run(vals)
ref = kernel.reference(vals.cpu()).cuda()
mismatch = (out != ref).any(dim=1)
n_bad = int(mismatch.sum())
ex = []
if n_bad:
    idxs = mismatch.nonzero().flatten()[:5]
    ex = [[int(vals[i]), int(out[i,0]), int(out[i,1]), int(ref[i,0]), int(ref[i,1])] for i in idxs]
print(json.dumps({"n_tested": int(vals.numel()), "n_large_gt_2p31": int((vals.cpu().to(torch.int64) > 2**31).sum()), "n_mismatch": n_bad, "examples_val_outrow_outcol_refrow_refcol": ex, "match": n_bad == 0, "triton_version": __import__("triton").__version__}))
