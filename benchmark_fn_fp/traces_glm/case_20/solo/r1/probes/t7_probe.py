
import torch, json, sys
sys.path.insert(0, "/root/cases/case_20")
from kernel import layer_norm

torch.manual_seed(0)
dev = 'cuda'
N = 512
# row 0: standard random; row 1: small variance; row 2: constant row
x = torch.randn(2, N, device=dev)
x_small = (torch.full((N,), 2.0, device=dev) + torch.randn(N, device=dev) * 1e-4).unsqueeze(0)
x = torch.cat([x, x_small, torch.full((1, N), 3.0, device=dev)], dim=0)
w = torch.randn(N, device=dev)
b = torch.randn(N, device=dev)
eps = 1e-5

y = layer_norm(x.float(), w.float(), b.float(), eps)

def ref(xr):
    mean = xr.mean(-1, keepdim=True)
    var = ((xr - mean) ** 2).mean(-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    return (xr - mean) * rstd * w + b

yr = ref(x.float())
diff = (y - yr).abs()
rowwise = diff.max(dim=-1).values

# expected relative error for small-variance row: rstd_kernel/rstd_ref
var_small = ((x_small - x_small.mean())**2).mean()
ratio = 1.0/(var_small.sqrt() + eps) / (1.0/(var_small + eps).sqrt())
print(json.dumps({
    "max_abs_err": diff.max().item(),
    "rowwise_max_abs_err": rowwise.tolist(),
    "small_var_row_var": var_small.item(),
    "small_var_rstd_ratio_kernel_over_ref": ratio.item(),
    "constant_row_out": y[3, :3].tolist(),
    "ref_constant_row_out": yr[3, :3].tolist(),
}))
