
import json, sys, torch
sys.path.insert(0, "/root/numerical_cases/case_g")
import kernel

values, = kernel.make_inputs("cuda")
out = kernel.run(values)
torch.cuda.synchronize()
ref = values.to(torch.float64).sum(dim=1)
den = max(float(ref.norm()), 0.008)
E = float((out.to(torch.float64) - ref).norm()) / den
finite = bool(torch.isfinite(out).all())
print(json.dumps({"metric": "relative L2 error E vs float64 reference", "E": E, "contract_max": 0.1,
                  "finite": finite, "ref_norm": float(ref.norm()),
                  "out_norm": float(out.to(torch.float64).norm()),
                  "max_abs_err": float((out.to(torch.float64)-ref).abs().max()),
                  "ref_min": float(ref.min()), "ref_max": float(ref.max())}))
