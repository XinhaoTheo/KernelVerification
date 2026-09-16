
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_27/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = "cuda"

def ref_spec(x, mask):
    neg = torch.full_like(x, -float("inf"))
    mx = torch.where(mask, x, neg).max(1, keepdim=True).values
    e = torch.where(mask, torch.exp(x - mx), torch.zeros_like(x))
    den = e.sum(1, keepdim=True)
    return torch.where(den > 0, e / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(e))

results = {}
worst = 0.0
bad_drop = 0
nonfinite = 0
torch.manual_seed(1)
for (R, C) in [(4,5),(8,7),(16,33),(3,1),(5,64),(6,129),(7,1000)]:
    x = (torch.randn(R, C, device=dev, dtype=torch.float32) * 5.0)
    mask = torch.rand(R, C, device=dev) > 0.4
    # force every row to keep >= 1 position
    idx = torch.randint(0, C, (R,), device=dev)
    mask[torch.arange(R, device=dev), idx] = True
    y = m.masked_softmax(x, mask); torch.cuda.synchronize()
    r = ref_spec(x, mask)
    err = float((y - r).abs().max())
    dz = int((y[~mask] != 0).sum())
    nf = int((~torch.isfinite(y)).sum())
    rowsum = y.sum(1)
    results[f"{R}x{C}"] = {"max_abs_err": err, "dropped_nonzero": dz, "nonfinite": nf,
                            "max_rowsum_dev": float((rowsum - 1).abs().max())}
    worst = max(worst, err); bad_drop += dz; nonfinite += nf

# adversarial: identical logits, very negative logits, ties
x = torch.full((3, 16), -1e30, device=dev); x[1] = 0.0; x[2] = 1e2
mask = torch.zeros(3, 16, dtype=torch.bool, device=dev); mask[:, :4] = True
y = m.masked_softmax(x, mask); torch.cuda.synchronize()
r = ref_spec(x, mask)
extreme = {"max_abs_err": float((y - r).abs().max()),
           "nonfinite": int((~torch.isfinite(y)).sum()),
           "dropped_nonzero": int((y[~mask] != 0).sum()),
           "row_examples": [[float(v) for v in y[i, :5].tolist()] for i in range(3)]}

print(json.dumps({"metric": "max abs err vs spec reference on rows keeping >=1 position; dropped entries exactly 0",
                  "per_shape": results, "worst_max_abs_err": worst,
                  "total_dropped_nonzero": bad_drop, "total_nonfinite": nonfinite,
                  "extreme_case": extreme}))
