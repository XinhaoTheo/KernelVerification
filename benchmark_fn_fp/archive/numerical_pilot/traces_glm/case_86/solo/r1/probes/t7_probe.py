import torch, numpy as np, json, sys
sys.path.insert(0, '/root/pilot_cases/case_86')
import kernel as km

x, w = km.make_inputs("cuda")
out = km.run(x, w)
ref = w.to(torch.float64) @ x.to(torch.float64)
num = torch.norm(out.to(torch.float64).flatten() - ref.flatten()).item()
den = max(torch.norm(ref.flatten()).item(), 0.001*np.sqrt(ref.numel()))
print(json.dumps({"E": num/den, "finite": bool(torch.isfinite(out).all().item()), "num": num, "den": den, "shape": list(out.shape)}))