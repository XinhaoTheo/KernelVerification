
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(x):
    absmax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale

torch.manual_seed(0)
x = (torch.randn(4, 256, device="cuda") * 10).contiguous()
out = k.quant_dequant_int8(x)
r = ref(x)
err = (out - r).abs()
row_absmax = x.abs().amax(dim=1)
res = {
  "shape": list(x.shape),
  "row_absmax": [round(v,4) for v in row_absmax.tolist()],
  "kernel_out_absmax_per_row": [round(v,4) for v in out.abs().amax(dim=1).tolist()],
  "ref_out_absmax_per_row": [round(v,4) for v in r.abs().amax(dim=1).tolist()],
  "max_abs_err": float(err.max()),
  "allowed_step_half_per_row": [round(v/254,6) for v in row_absmax.tolist()],
  "n_clipped_at_4": int((out.abs() >= 4.0 - 1e-6).sum()),
  "n_input_gt_4": int((x.abs() > 4.0).sum()),
  "allclose_rtol1e-2_atol1e-2": bool(torch.allclose(out, r, rtol=1e-2, atol=1e-2)),
  "metric": "max abs err vs contract reference; clipping count",
}
print(json.dumps(res))
