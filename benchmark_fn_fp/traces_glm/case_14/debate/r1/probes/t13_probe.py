import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_14/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(x):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale

torch.manual_seed(1)
# low-magnitude rows, absmax ~0.05
x = torch.randn(8, 128, device="cuda") * 0.02
out = m.quant_dequant_int8(x)
r = ref(x)
err = (out - r).abs()
row_absmax = x.abs().amax(dim=-1)
contract_bound = (row_absmax / 127 / 2)  # <= scale/2 rounding bound
print(json.dumps({
  "metric": "max_abs_err_kernel_vs_contract_reference",
  "input_row_absmax": [float(a) for a in row_absmax],
  "max_abs_err": float(err.max()),
  "contract_scale_half_bound": float(contract_bound.max()),
  "err_exceeds_contract_bound": bool((err.amax(dim=-1) > contract_bound).all()),
  "ratio_err_to_bound": float((err.amax(dim=-1) / contract_bound).min()),
}))