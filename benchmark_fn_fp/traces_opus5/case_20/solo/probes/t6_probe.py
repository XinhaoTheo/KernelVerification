
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_20/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'
eps=1e-5
N=512
x = torch.randn(8, N, device=dev, dtype=torch.float32)
# make row 0 constant, row 1 near-constant (tiny variance)
x[0] = 3.7
x[1] = 3.7 + 1e-6*torch.randn(N, device=dev)
w = torch.randn(N, device=dev); b = torch.randn(N, device=dev)

y = m.layer_norm(x, w, b, eps)
mean = x.mean(-1, keepdim=True)
var = ((x-mean)**2).mean(-1, keepdim=True)
ref = (x-mean)*torch.rsqrt(var+eps)*w + b
ref_t = torch.nn.functional.layer_norm(x, (N,), w, b, eps)

diff = (y-ref).abs()
res = {
 "metric":"max abs / rel error per row vs contract formula rstd=1/sqrt(var+eps)",
 "row_var": var.flatten().tolist(),
 "row_max_abs_err": diff.max(dim=1).values.tolist(),
 "row_max_rel_err": (diff/(ref.abs()+1e-6)).max(dim=1).values.tolist(),
 "max_abs_err_vs_torch_LN": (y-ref_t).abs().max().item(),
 "constant_row_y_absmax": y[0].abs().max().item(),
 "constant_row_ref_absmax": ref[0].abs().max().item(),
 "kernel_rstd_row0_implied": None,
}
# implied rstd row0
d0 = (x[0]-x[0].mean())
res["kernel_rstd_row0_expected_wrong"] = 1.0/(var[0].item()**0.5+eps)
res["contract_rstd_row0"] = 1.0/((var[0].item()+eps)**0.5)
# normal row check ratio y/ref deviation
r = ((y[3]-b)/((ref[3]-b)+1e-12))
res["row3_ratio_kernel_over_contract_median"] = r.median().item()
print(json.dumps(res, indent=1))
