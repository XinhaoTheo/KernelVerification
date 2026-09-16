
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}

def check(probs, k, tag):
    p = probs.cuda().contiguous().float()
    out = m.topk_mask(p, k)
    kth = p.topk(k, dim=1).values[:, -1:]
    ref = torch.where(p >= kth, p, torch.zeros_like(p))
    kept = out != 0
    cnt = kept.sum(dim=1)
    bad = kept & (p < kth)
    # renormalized downstream sampling weights
    w_out = out / out.sum(dim=1, keepdim=True)
    w_ref = ref / ref.sum(dim=1, keepdim=True)
    tv = 0.5 * (w_out - w_ref).abs().sum(dim=1)
    spurious_mass = (w_out * bad).sum(dim=1)
    res[tag] = dict(k=k, N=p.shape[1],
                    kept_count_min=int(cnt.min()), kept_count_max=int(cnt.max()),
                    kept_strictly_below_kth=int(bad.sum()),
                    max_TV_vs_reference=float(tv.max()),
                    max_spurious_sampling_mass=float(spurious_mass.max()),
                    max_prob=float(p.max()), kth_prob_row0=float(kth[0]))

V = 32000
for T in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]:
    logits = torch.randn(4, V) * T
    check(torch.softmax(logits, dim=1), 50, f"softmax_logitstd{T:g}_k50")
    check(torch.softmax(logits, dim=1), 5, f"softmax_logitstd{T:g}_k5")

# realistic zipf-like LM distribution
idx = torch.arange(1, V+1, dtype=torch.float32)
for a in [1.0, 2.0, 3.0]:
    p = (idx ** (-a)).repeat(4,1)
    p = p / p.sum(dim=1, keepdim=True)
    check(p, 50, f"zipf_a{a:g}_k50")
print(json.dumps(res, indent=1))
