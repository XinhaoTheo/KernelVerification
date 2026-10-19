
import torch, json, sys
sys.path.insert(0, "/root/cases/case_25")
from kernel import requantize

torch.manual_seed(0)
N = 4096
x = torch.randn(N, device="cuda", dtype=torch.float32)

def ref(x, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    for i in range(0, n, BLOCK):
        blk = x[i:i+BLOCK]
        absmax = blk.abs().max()
        scale = torch.where(absmax == 0, torch.tensor(1.0, device=x.device), absmax/127.0)
        q = torch.round(blk/scale).clamp(-127,127)
        out[i:i+BLOCK] = q*scale
    return out

y = requantize(x)
r = ref(x)
scale0 = x[:1024].abs().max().item()/127.0

res = {}
res["mean_signed_err_kernel_vs_input"] = (y-x).mean().item()
res["mean_signed_err_ref_vs_input"] = (r-x).mean().item()
res["scale_block0"] = scale0
res["mean_err_in_lsb_kernel"] = ((y-x)[:1024]/scale0).mean().item()
res["mean_err_in_lsb_ref"] = ((r-x)[:1024]/scale0).mean().item()
res["max_abs_kernel_vs_ref"] = (y-r).abs().max().item()
res["frac_mismatch_kernel_vs_ref"] = (y!=r).float().mean().item()

# repeated application drift
a = x.clone(); b = x.clone()
drift_k, drift_r = [], []
for step in range(20):
    a = requantize(a); b = ref(b)
    drift_k.append(a.mean().item()); drift_r.append(b.mean().item())
res["input_mean"] = x.mean().item()
res["kernel_mean_after_1_5_20"] = [drift_k[0], drift_k[4], drift_k[19]]
res["ref_mean_after_1_5_20"] = [drift_r[0], drift_r[4], drift_r[19]]
res["kernel_absmean_after_20"] = a.abs().mean().item()
res["ref_absmean_after_20"] = b.abs().mean().item()
res["input_absmean"] = x.abs().mean().item()
res["metric"] = "mean signed error in LSB units vs round-to-nearest reference; floor predicts -0.5 LSB"
print(json.dumps(res, indent=2))
