import torch, json, sys
sys.path.insert(0, "/root/cases/case_166")
import kernel as K

xs = [65504.0, 32768.0, 16384.0, 8192.0, 60000.0, 40000.0, 65504.0*0.999]
x = torch.tensor(xs, dtype=torch.float16).to("cuda")
out = K.run(x)
ref = torch.log1p(x.to(torch.float64)).to("cuda")
err = (out.to(torch.float64) - ref).abs()
res = {
 "metric": "per-value abs error vs FP64 log1p for large FP16 inputs (exponent 13-16)",
 "inputs": xs,
 "errors": err.tolist(),
 "max_abs_err": err.max().item(),
 "tolerance": 1e-6,
 "all_within_tol": bool((err <= 1e-6).all().item()),
}
print(json.dumps(res))
