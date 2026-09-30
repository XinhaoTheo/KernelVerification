import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_d")
import kernel

logits_np, values_np = kernel.make_inputs_numpy()
logits = torch.from_numpy(logits_np).cuda()
values = torch.from_numpy(values_np).cuda()
out = kernel.run(logits, values).cpu().numpy().astype(np.float64)

l = logits_np.astype(np.float64)
v = values_np.astype(np.float64)
w = np.exp(l - l.max())
p = w / w.sum()
ref = p @ v

denom = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
err = np.linalg.norm(out - ref) / denom
print(json.dumps({"rel_l2": float(err), "tol": 0.02, "finite": bool(np.all(np.isfinite(out))),
                  "ref_norm": float(np.linalg.norm(ref)), "out_norm": float(np.linalg.norm(out))}))