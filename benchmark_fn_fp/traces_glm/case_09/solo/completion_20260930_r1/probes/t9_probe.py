
import torch, json, sys, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_09/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
def ref_lowest_tied(row):
    return int(torch.arange(row.numel())[row == row.max()].min())
res = {}
for N in [2,4,8,16,32,64]:
    B = 2000
    torch.manual_seed(0)
    bad = 0; ex = None
    # random rows, then set the max value at multiple random positions to create ties
    scores = torch.randn(B, N, device=dev)
    for b in range(B):
        idxs = torch.randperm(N)[:2 + (b % (N-1))][:2].cuda()  # pick 2 tied positions
        scores[b, idxs[0]] = 5.0; scores[b, idxs[1]] = 5.0
        # sometimes leave another position as unique second
    out = k.sorted_topk_indices(scores, 1).squeeze(-1)
    for b in range(B):
        lo = ref_lowest_tied(scores[b])
        if out[b].item() != lo:
            bad += 1
            if ex is None: ex = (b, scores[b].tolist(), out[b].item(), lo)
    # also all-equal rows
    scores_eq = torch.zeros(100, N, device=dev)
    out_eq = k.sorted_topk_indices(scores_eq, 1).squeeze(-1)
    eq_bad = (out_eq != 0).sum().item()
    res[f"N{N}"] = {"tied_rows_bad": bad, "total": B, "all_equal_bad_nonzero": eq_bad, "example": ex}
print(json.dumps(res))
