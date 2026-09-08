
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_28/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(7)
dev="cuda"; R,C = 16, 768

# realistic transformer activation: per-channel lognormal scale, heavy tail
chan_scale = torch.exp(torch.randn(C, device=dev)*0.6)
# a few genuine outlier channels at ~10-20x the bulk
out_cols = torch.randperm(C, device=dev)[:4]
chan_scale[out_cols] *= torch.tensor([10.0,13.0,17.0,20.0], device=dev)
x = torch.randn(R, C, device=dev) * chan_scale

y = k.quant_dequant(x)
def rel(y,x): return ((y-x).norm(dim=1)/x.norm(dim=1))
rk = rel(y,x)

s = x.abs().amax(dim=1, keepdim=True)/127.0
yref = torch.clamp(torch.round(x/s), -127, 127)*s
rr = rel(yref, x)

ratio = (y.abs().amax(dim=1)/x.abs().amax(dim=1))
out = {
 "shape": list(x.shape), "pctl_levels": 2, "tolerance": 0.05,
 "kernel_rel_l2_max": round(rk.max().item(),5),
 "kernel_rel_l2_median": round(rk.median().item(),5),
 "kernel_rel_l2_min": round(rk.min().item(),5),
 "absmax_ref_rel_l2_max": round(rr.max().item(),6),
 "rows_kernel_exceeding_tol": int((rk>0.05).sum().item()),
 "rows_ref_exceeding_tol": int((rr>0.05).sum().item()),
 "rows_total": R,
 "min_maxabs_out_over_in": round(ratio.min().item(),5),
 "max_outlier_to_median_absratio_row0": round((x[0].abs().max()/x[0].abs().median()).item(),3),
}
print(json.dumps(out))
