
import json, sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_25/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
dev = "cuda"
x = torch.randn(4096, device=dev, dtype=torch.float32)

def ref(x, BLOCK=1024):
    y = x.clone()
    n = x.numel()
    for i in range(0, n, BLOCK):
        b = x[i:i+BLOCK]
        am = b.abs().max()
        s = torch.where(am == 0, torch.tensor(1.0, device=b.device), am/127.0)
        y[i:i+BLOCK] = torch.round(b/s)*s
    return y

out = k.requantize(x)
r = ref(x)

# per-block scale (all blocks 1024)
scales = [float(x[i:i+1024].abs().max()/127.0) for i in range(0,4096,1024)]
mean_scale = sum(scales)/len(scales)

err_kernel = (out - x)
err_ref = (r - x)
# normalize error by per-element block scale
sc = torch.empty_like(x)
for j,i in enumerate(range(0,4096,1024)):
    sc[i:i+1024] = scales[j]

res = {
 "mean_signed_err_kernel_in_LSB": float((err_kernel/sc).mean()),
 "mean_signed_err_ref_in_LSB": float((err_ref/sc).mean()),
 "max_abs_err_kernel_in_LSB": float((err_kernel/sc).abs().max()),
 "max_abs_err_ref_in_LSB": float((err_ref/sc).abs().max()),
 "max_abs_diff_kernel_vs_ref": float((out-r).abs().max()),
 "frac_elems_differ_from_ref": float(((out-r).abs() > 1e-9).float().mean()),
 "mean_scale": mean_scale,
}

# repeated application drift
z = x.clone(); means_k = [float(z.mean())]
for _ in range(10):
    z = k.requantize(z); means_k.append(float(z.mean()))
w = x.clone(); means_r = [float(w.mean())]
for _ in range(10):
    w = ref(w); means_r.append(float(w.mean()))
res["kernel_mean_trajectory"] = means_k
res["ref_mean_trajectory"] = means_r
res["kernel_mean_drift_10_steps"] = means_k[-1]-means_k[0]
res["ref_mean_drift_10_steps"] = means_r[-1]-means_r[0]
res["kernel_monotone_decreasing"] = all(means_k[i+1] <= means_k[i]+1e-12 for i in range(10))
print(json.dumps(res))
