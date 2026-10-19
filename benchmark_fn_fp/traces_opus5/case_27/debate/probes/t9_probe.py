
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_28/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
dev = "cuda"
R, C = 8, 512
x = torch.randn(R, C, device=dev, dtype=torch.float32)
# inject 3 heavy-tail outlier channels per row at 30x-100x the bulk
for r in range(R):
    cols = torch.randperm(C, device=dev)[:3]
    mags = torch.tensor([30.0, 60.0, 100.0], device=dev)
    signs = torch.tensor([1.0, -1.0, 1.0], device=dev)
    x[r, cols] = mags * signs

y = k.quant_dequant(x)   # default pctl_levels=2

def rel(y, x):
    return ((y - x).norm(dim=1) / x.norm(dim=1))

rel_k = rel(y, x)

# absmax reference (contract-correct calibration)
scale_ref = x.abs().amax(dim=1, keepdim=True) / 127.0
q_ref = torch.clamp(torch.round(x / scale_ref), -127, 127)
y_ref = q_ref * scale_ref
rel_ref = rel(y_ref, x)

# how many entries got clamped by the kernel
ax = x.abs()
srt = ax.sort(dim=1, descending=True).values
out = {
  "shape": list(x.shape), "dtype": "float32", "pctl_levels": 2,
  "kernel_rel_l2_per_row": [round(v,5) for v in rel_k.tolist()],
  "kernel_rel_l2_max": round(rel_k.max().item(),5),
  "absmax_ref_rel_l2_max": round(rel_ref.max().item(),6),
  "tolerance": 0.05,
  "rows_exceeding_tolerance": int((rel_k > 0.05).sum().item()),
  "max_abs_out_over_max_abs_in_row0": round((y[0].abs().max()/x[0].abs().max()).item(),5),
  "top3_absmag_row0": [round(v,3) for v in srt[0,:3].tolist()],
}
print(json.dumps(out))
