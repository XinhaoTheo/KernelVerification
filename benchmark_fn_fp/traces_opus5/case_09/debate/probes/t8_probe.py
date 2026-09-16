
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k9", "/root/cases/case_09/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}
overall = {"argmax_mismatch_rows": 0, "invalid_perm_rows": 0, "inversion_rows": 0, "total_rows": 0}
err = None
for nd in range(1, 11):
    N = 2**nd
    B = 64
    # strictly distinct values per row: random permutation of a distinct grid, scaled/shifted per row
    base = torch.randn(B, N, device="cuda", dtype=torch.float32)
    # enforce distinctness by ranking and rebuilding from a distinct grid
    order = base.argsort(dim=1)
    grid = torch.linspace(-3.0, 3.0, N, device="cuda", dtype=torch.float32)
    scores = torch.empty(B, N, device="cuda", dtype=torch.float32)
    scores.scatter_(1, order, grid.unsqueeze(0).expand(B, N))
    assert all(len(torch.unique(scores[i])) == N for i in range(B))
    try:
        top1 = m.sorted_topk_indices(scores, 1).squeeze(1)
        full = m.sorted_topk_indices(scores, N)
    except Exception as e:
        err = f"N={N}: {type(e).__name__}: {e}"
        res[str(N)] = {"error": err}
        break
    ref = scores.argmax(dim=1)
    mism = int((top1 != ref).sum())
    # permutation validity
    srt = full.sort(dim=1).values
    ident = torch.arange(N, device="cuda").unsqueeze(0).expand(B, N)
    bad_perm = int((srt != ident).any(dim=1).sum())
    gath = scores.gather(1, full)
    inv = int((gath[:, 1:] > gath[:, :-1]).any(dim=1).sum())
    res[str(N)] = {"argmax_mismatch_rows": mism, "invalid_perm_rows": bad_perm,
                   "inversion_rows": inv, "rows": B,
                   "example_top1": int(top1[0]), "example_ref": int(ref[0])}
    overall["argmax_mismatch_rows"] += mism
    overall["invalid_perm_rows"] += bad_perm
    overall["inversion_rows"] += inv
    overall["total_rows"] += B

out = {"per_N": res, "overall": overall, "error": err,
       "metric": "k=1 index vs torch.argmax; full-row permutation validity; non-increasing gathered scores",
       "input": "strictly distinct float32 rows (permuted linspace(-3,3,N)), B=64, N=2..1024"}
print(json.dumps(out))
