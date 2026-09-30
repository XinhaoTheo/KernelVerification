
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_09/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
res = {}
for N in [2,4,8,16,32,64]:
    B = 2000
    torch.manual_seed(0)
    scores = torch.randn(B, N, device=dev)
    ar = torch.arange(N, device=dev)
    for b in range(B):
        p = torch.randperm(N)[:2].to(dev)
        scores[b, p] = 5.0
    out = k.sorted_topk_indices(scores, 1).squeeze(-1)
    # reference: lowest index among tied max
    lo = (scores == scores.max(dim=1, keepdim=True).values).float() * ar[None, :]
    lo[lo == 0] = N
    lo = lo.argmin(dim=1)
    bad = int((out != lo).sum().item())
    ex = None
    if bad:
        m = (out != lo).nonzero()[0]
        b = int(m[0]); ex = {"b": b, "scores": scores[b].tolist(), "kernel": int(out[b]), "ref": int(lo[b])}
    # all-equal rows: correct answer is index 0
    scores_eq = torch.zeros(100, N, device=dev)
    out_eq = k.sorted_topk_indices(scores_eq, 1).squeeze(-1)
    eq_bad = int((out_eq != 0).sum().item())
    res[f"N{N}"] = {"tied_rows_bad": bad, "total": B, "all_equal_bad_nonzero": eq_bad, "example": ex}
print(json.dumps(res))
