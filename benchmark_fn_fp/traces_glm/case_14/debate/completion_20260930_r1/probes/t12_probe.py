import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(x):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale

torch.manual_seed(0)
# rows with absmax > 4.0
x = torch.randn(8, 128, device="cuda") * 3.0
x[0, 0] = 10.0
out = m.quant_dequant_int8(x)
r = ref(x)
err = (out - r).abs()
print(json.dumps({
  "metric": "max_abs_err_kernel_vs_contract_reference",
  "kernel_max_out": float(out.abs().max()),
  "kernel_row0_saturates_at_4": float(out[0].abs().max()),
  "ref_row0_saturates_at": float(r[0].abs().max()),
  "max_abs_err": float(err.max()),
  "input_row_absmax": [float(a) for a in x.abs().amax(dim=-1)],
}))