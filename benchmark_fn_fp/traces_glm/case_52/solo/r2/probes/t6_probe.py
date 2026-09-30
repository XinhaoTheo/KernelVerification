import sys, json, torch, numpy as np
sys.path.insert(0, "/root/numerical_cases/case_q")
import kernel

q, c, e = kernel.make_inputs("cuda")
out = kernel.run(q, c, e)

# float64 reference
d = ((c.double() - q.double())**2).sum(dim=1)
ref_idx = int(torch.argmin(d))
ref = e[ref_idx].double()
rel = ((out.double() - ref).norm() / max(ref.norm().item(), 1e-12)).item()

# quantized winner index (simulate kernel logic)
qf = q.float(); cf = c.float()
qq = torch.floor(qf*8+0.5)*0.125
cq = torch.floor(cf*8+0.5)*0.125
qd = ((cq-qq)**2).float().sum(dim=1)
kern_idx = int(torch.argmin(qd))

print(json.dumps({
  "kernel_output": out.cpu().tolist(),
  "reference_idx": ref_idx, "quantized_idx": kern_idx,
  "ref_distances": d.cpu().tolist(),
  "quant_distances": qd.cpu().tolist(),
  "relative_l2_error": rel,
  "passes_0.1": rel <= 0.1,
  "output_finite": bool(torch.isfinite(out).all())
}))