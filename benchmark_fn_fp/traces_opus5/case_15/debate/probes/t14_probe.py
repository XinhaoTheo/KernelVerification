
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_15/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

sc = 0.125  # exactly representable; absmax = 127*sc = 15.875
vals = []
for m in range(32):
    vals.append((m + 0.5) * sc)
    vals.append(-(m + 0.5) * sc)
vals = vals[:63]
vals.append(127 * sc)  # absmax element so that scale == 0.125 exactly
x = torch.tensor([vals], device="cuda", dtype=torch.float32)
assert tuple(x.shape) == (1, 64), tuple(x.shape)

am = x.abs().amax(dim=1, keepdim=True)
scale = am / 127.0
step = scale.item()
ratio = x / scale
half_even = torch.round(ratio).clamp(-127, 127) * scale
half_away = torch.where(ratio >= 0, torch.floor(ratio + 0.5), torch.ceil(ratio - 0.5)).clamp(-127, 127) * scale
half_up = torch.floor(ratio + 0.5).clamp(-127, 127) * scale

out = k.group_quant_dequant(x, 64)

def cnt(a, b):
    return int(((a - b).abs() > 0.25 * step).sum().item())

d_even = (out - half_even).abs()
d_away = (out - half_away).abs()
pos = x > 0
# how many ratios are exactly on a .5 tick
frac = (ratio - torch.floor(ratio)).abs()
exact_ties = int((frac == 0.5).sum().item())

print(json.dumps({
  "scale": step,
  "scale_is_exactly_0.125": abs(step - 0.125) == 0.0,
  "exact_tie_count": exact_ties,
  "n_elems": int(x.numel()),
  "mismatch_vs_torch_round_half_even": cnt(out, half_even),
  "mismatch_vs_half_away_from_zero": cnt(out, half_away),
  "mismatch_vs_half_up_floor_kernel_semantics": cnt(out, half_up),
  "max_abs_err_vs_half_even": d_even.max().item(),
  "max_err_in_steps_vs_half_even": d_even.max().item() / step,
  "max_abs_err_vs_half_away": d_away.max().item(),
  "mismatch_positive_ties_vs_half_even": int(((d_even > 0.25 * step) & pos).sum().item()),
  "mismatch_negative_ties_vs_half_even": int(((d_even > 0.25 * step) & ~pos).sum().item()),
  "mismatch_negative_ties_vs_half_away": int(((d_away > 0.25 * step) & ~pos).sum().item()),
  "sample_out_first8": out[0, :8].tolist(),
  "sample_half_even_first8": half_even[0, :8].tolist(),
  "sample_half_away_first8": half_away[0, :8].tolist()
}))
