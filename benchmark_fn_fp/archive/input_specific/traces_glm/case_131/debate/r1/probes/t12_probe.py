import json, math, numpy as np, torch, sys
sys.path.insert(0, "/root/evidence_cases/case_e13")
import kernel

x, = kernel.make_inputs()
before = x.clone()
out = kernel.run(x)
xn = x.detach().cpu().numpy()
vals = xn.astype(np.float64)
target = [math.fsum(vals[r].tolist()) for r in range(4)]
actual = out.detach().cpu().numpy().astype(np.float64)
err = float(np.linalg.norm(actual - np.array(target)) / max(np.linalg.norm(np.array(target)), 1e-12))
result = {
 "target_exact_fsum": target,
 "output": actual.tolist(),
 "relative_error_exact": err,
 "tolerance": 1e-5,
 "structural_ok": bool(tuple(out.shape)==(4,) and out.dtype==torch.float32 and np.isfinite(actual).all() and torch.equal(x, before)),
 "passes_contract_metric": bool(err <= 1e-5),
}
print(json.dumps(result))