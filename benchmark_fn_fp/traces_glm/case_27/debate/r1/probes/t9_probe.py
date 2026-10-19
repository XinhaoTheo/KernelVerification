import torch, json, sys
sys.path.insert(0, "/root/cases/case_28")
from kernel import quant_dequant

torch.manual_seed(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# Row where top-3 |x| are tied: bulk small (~0.1 magnitude), three channels tied at 0.5
x = torch.full((4, 1024), 0.05, device=dev) + 0.02 * torch.randn(4, 1024, device=dev)
x[:, 0] = 0.5; x[:, 1] = -0.5; x[:, 2] = 0.5
y = quant_dequant(x)
per_row = ((y - x).norm(dim=1) / x.norm(dim=1))
# reference: absmax calibration
s = x.abs().amax(dim=1, keepdim=True) / 127.0
q = torch.clamp(torch.round(x / s), -127, 127)
y_ref = q * s
ref_err = ((y_ref - x).norm(dim=1) / x.norm(dim=1))
print(json.dumps({
    "metric": "per_row_relative_L2_error, kernel vs absmax-reference",
    "kernel_per_row_err": [round(v, 4) for v in per_row.tolist()],
    "max_kernel_err": round(per_row.max().item(), 4),
    "absmax_reference_max_err": round(ref_err.max().item(), 4),
    "kernel_y_zero_fraction": float((y == 0).float().mean()),
    "exceeds_5pct": bool((per_row > 0.05).any()),
}))
