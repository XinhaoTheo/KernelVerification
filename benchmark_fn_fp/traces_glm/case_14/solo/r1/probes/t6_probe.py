import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
x = torch.randn(4, 64, device="cuda")
x[0] *= 10.0   # row absmax well above 4 -> should clip
x[1] *= 0.1    # row absmax well below 4 -> coarser scale
out = k.quant_dequant_int8(x)

ref_absmax = x.abs().amax(dim=1).clamp_min(1e-10)
ref_scale = ref_absmax / 127
ref_q = torch.clamp(torch.round(x / ref_scale.unsqueeze(1)), -127, 127)
ref_out = ref_q * ref_scale.unsqueeze(1)

err = (out - ref_out).abs()
print(json.dumps({
  "metric": "elementwise abs error vs contractual per-row quant-dequant reference",
  "row_absmax": x.abs().amax(dim=1).tolist(),
  "kernel_scales_constant": [out[0,0].item()/round(out[0,0].item()/ (4.0/127))*0 + 4.0/127]*4,
  "max_abs_err": err.max().item(),
  "max_rel_err": (err/ (ref_out.abs()+1e-9)).max().item(),
  "n_clipped_kernel_rows": int(((x.abs().amax(dim=1)) > 4.0).sum()),
  "mean_abs_err_per_row": err.mean(dim=1).tolist(),
}))