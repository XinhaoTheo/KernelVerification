
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_15/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

sc = 1.0/8.0   # exactly representable; absmax = 127*sc
vals = []
# ties at +/- (m + 0.5)*sc for m = 0..30
for m in range(31):
    vals.append((m+0.5)*sc)
    vals.append(-(m+0.5)*sc)
vals = vals[:63]
vals.append(127*sc)   # set absmax so scale == sc exactly
x = torch.tensor([vals], device="cuda", dtype=torch.float32)
assert x.shape == (1,64)

am = x.abs().amax(dim=1, keepdim=True)
scale = am/127.0
ratio = x/scale
half_even = torch.round(ratio).clamp(-127,127)*scale          # torch.round = half-to-even
half_away = torch.where(ratio>=0, torch.floor(ratio+0.5), torch.ceil(ratio-0.5)).clamp(-127,127)*scale
half_up   = torch.floor(ratio+0.5).clamp(-127,127)*scale      # kernel semantics

out = k.group_quant_dequant(x, 64)
step = scale.item()
def cnt(a,b): return int(((a-b).abs() > 0.25*step).sum().item())
d_even = (out-half_even).abs()
pos_mask = x > 0
print(json.dumps({
 "scale": step,
 "exact_scale_matches_1/8": abs(step-0.125) < 1e-9,
 "n_ties": 63,
 "mismatch_vs_torch_round_half_even": cnt(out, half_even),
 "mismatch_vs_half_away_from_zero": cnt(out, half_away),
 "mismatch_vs_half_up_floor": cnt(out, half_up),
 "max_abs_err_vs_half_even": d_even.max().item(),
 "one_step_err_ratio": (d_even.max().item()/step) if step>0 else None,
 "mismatch_positive_ties": int(((d_even > 0.25*step) & pos_mask).sum().item()),
 "mismatch_negative_ties": int(((d_even > 0.25*step) & ~pos_mask).sum().item())
}))
