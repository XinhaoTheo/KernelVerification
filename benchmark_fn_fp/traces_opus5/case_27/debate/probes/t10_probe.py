
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_28/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(1)
dev="cuda"; C=512

def rel(y,x): return ((y-x).norm(dim=1)/x.norm(dim=1))

# Case A: 3 DISTINCT outlier magnitudes
xa = torch.randn(1, C, device=dev)
xa[0, :3] = torch.tensor([50.0, 55.0, 60.0], device=dev)
# Case B: SAME outlier magnitude duplicated in 4 columns
xb = xa.clone()
xb[0, :3] = torch.randn(3, device=dev)      # remove distinct outliers
xb[0, 10:14] = torch.tensor([55.0,-55.0,55.0,-55.0], device=dev)

ya = k.quant_dequant(xa); yb = k.quant_dequant(xb)
ra = rel(ya, xa).item(); rb = rel(yb, xb).item()

def clamped_count(x,y):
    return int((~torch.isclose(y, x, rtol=0.02, atol=1e-3)).sum().item())

# infer effective thresh = max |y|
out = {
 "distinct_outliers_rel_l2": round(ra,5),
 "tied_outliers_rel_l2": round(rb,5),
 "tolerance": 0.05,
 "distinct_max_abs_out": round(ya.abs().max().item(),5),
 "tied_max_abs_out": round(yb.abs().max().item(),5),
 "distinct_max_abs_in": round(xa.abs().max().item(),5),
 "tied_max_abs_in": round(xb.abs().max().item(),5),
 "tied_worse_than_distinct": bool(rb > ra),
 "n_tied_outlier_cols": 4,
}
print(json.dumps(out))
