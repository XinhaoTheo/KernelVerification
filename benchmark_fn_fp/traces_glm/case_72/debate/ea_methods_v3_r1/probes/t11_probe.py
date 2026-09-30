import importlib.util, json
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

smoke, W, b, c = k.make_inputs_numpy()

# legal corner inputs: x = sign(W[k]) for each of the 6 rows, entries +-1 (in [-1,1])
X = np.stack([np.sign(W[j]).astype(np.float32) for j in range(6)])
args = [torch.from_numpy(a).cuda() for a in (X, W, b, c)]
before = [v.clone() for v in args]
actual = k.run(*args)  # on GPU
Xd = args[0].cpu().double()
Wd, bd, cd = [torch.from_numpy(a).double() for a in (W, b, c)]  # reference on CPU in float64
expected = 0.25*Xd[:,0] + 0.5*Xd[:,1] + torch.relu(Xd @ Wd.T - bd) @ cd
err = (actual.cpu().double() - expected).abs()
print(json.dumps({
  "metric": "max absolute error vs float64 target on row-aligned +/-1 corners",
  "max_error": float(err.max()),
  "tolerance": 1.0,
  "per_corner_errors": [round(float(e), 6) for e in err],
  "per_corner_residual": [round(float(r), 6) for r in (torch.relu(Xd @ Wd.T - bd) @ cd)],
  "legal_entries_in_box": bool(np.all(np.abs(X) <= 1)),
  "struct_ok": bool(actual.dtype == torch.float32 and actual.shape == (6,) and torch.isfinite(actual).all()),
  "inputs_unmodified": all(torch.equal(a, z) for a, z in zip(args, before)),
  "exceeds_1.0": bool(float(err.max()) > 1.0),
}))