
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1234)
res = {}

def check(p, k, tag):
    p = p.cuda().contiguous().float()
    assert bool(torch.isfinite(p).all()), "input must be finite"
    out = m.topk_mask(p, k)
    kth = p.topk(k, dim=1).values[:, -1:]
    ref = torch.where(p >= kth, p, torch.zeros_like(p))
    kept = out != 0
    bad = kept & (p < kth)
    nbad = int(bad.sum())
    w_out = out / out.sum(dim=1, keepdim=True)
    w_ref = ref / ref.sum(dim=1, keepdim=True)
    tv = 0.5*(w_out-w_ref).abs().sum(dim=1)
    res[tag] = dict(
        k=k, N=p.shape[1], all_input_finite=True,
        input_min=float(p.min()), input_max=float(p.max()),
        kept_count_min=int(kept.sum(1).min()), kept_count_max=int(kept.sum(1).max()),
        kept_strictly_below_kth=nbad,
        worst_kept_value=float(torch.where(bad, p, kth).min()) if nbad else None,
        cutoff_kth_row0=float(kth[0]),
        worst_kept_over_cutoff_ratio=float((torch.where(bad,p,kth)/kth).min()) if nbad else 1.0,
        exact_match_ref=bool(torch.equal(out, ref)),
        max_TV_downstream_weights=float(tv.max()),
        out_all_finite=bool(torch.isfinite(out).all()),
    )

V = 32000
check(torch.softmax(torch.randn(8, V)*20, dim=1), 50, "probs_logitstd20_k50")
check(torch.softmax(torch.randn(8, V)*3/0.1, dim=1), 50, "probs_std3_T0.1_k50")
check(torch.softmax(torch.randn(8, V)*30, dim=1), 5, "probs_logitstd30_k5")
# control: mild row must be exact
check(torch.softmax(torch.randn(8, V)*2, dim=1), 50, "control_logitstd2_k50")
print(json.dumps(res, indent=1))
