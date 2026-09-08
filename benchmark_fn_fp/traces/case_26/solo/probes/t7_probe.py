
import torch, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}
def check(scores, k, tag):
    out = m.topk_mask(scores, k)
    kth = scores.topk(k, dim=1).values[:, -1:]
    ref = torch.where(scores >= kth, scores, torch.zeros_like(scores))
    kept = out != 0
    # invariant: min kept >= max dropped
    big = torch.finfo(scores.dtype).max
    minkept = torch.where(kept, scores, torch.full_like(scores, big)).min(dim=1).values
    maxdrop = torch.where(~kept, scores, torch.full_like(scores, -big)).max(dim=1).values
    inv = bool((minkept >= maxdrop).all())
    cnt = kept.sum(dim=1)
    exact = bool(torch.equal(out, ref))
    # entries kept strictly below the k-th largest
    bad = (kept & (scores < kth)).sum().item()
    res[tag] = dict(exact_match_all_ties_ref=exact, invariant_kept_ge_dropped=inv,
                    min_count=int(cnt.min()), max_count=int(cnt.max()), k=k,
                    kept_below_kth=int(bad),
                    max_abs_diff=float((out-ref).abs().max()))

for tag, s in [("randn_64x1024", torch.randn(64,1024,device='cuda')),
               ("randn_8x128", torch.randn(8,128,device='cuda')),
               ("uniform_16x512", torch.rand(16,512,device='cuda')),
               ("softmaxlike", torch.softmax(torch.randn(16,512,device='cuda')*3,dim=1))]:
    for k in (1,5,50):
        if k <= s.shape[1]:
            check(s, k, f"{tag}_k{k}")
print(json.dumps(res, indent=1))
