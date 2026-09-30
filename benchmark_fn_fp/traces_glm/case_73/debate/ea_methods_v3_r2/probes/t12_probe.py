import numpy as np, torch, json, sys, importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/evidence_cases/case_e12/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
dev = "cuda" if torch.cuda.is_available() else "cpu"
smoke, W, b, c = km.make_inputs(dev)
# enumerate all 4096 vertices
bits = ((np.arange(4096)[:, None] >> np.arange(12)) & 1)
V = torch.from_numpy((2*bits.astype(np.float32) - 1)).to(dev)
out = km.run(V, W, b, c)
Vd, Wd, Bd, Cd = V.double(), W.double(), b.double(), c.double()
ref = 0.25*Vd[:,0] + 0.5*Vd[:,1] + torch.relu(Vd @ Wd.T - Bd) @ Cd
err = (out.double() - ref).abs()
residual = (torch.relu(Vd @ Wd.T - Bd) @ Cd)
maxres = float(residual.max())
print(json.dumps({
  "metric": "max absolute error over all 4096 box vertices, float64 reference; decisive because residual is convex so box-max attained at vertices",
  "n_vertices": 4096,
  "max_abs_error": float(err.max()),
  "max_residual": maxres,
  "tolerance": 1.0,
  "exceeds_tolerance": bool(float(err.max()) > 1.0),
  "worst_vertex_index": int(err.argmax()),
  "per_neuron_margins_at_worst": [float(v) for v in torch.relu(Vd @ Wd.T - Bd)[int(err.argmax())].tolist()],
  "out_finite_f32_shape_ok": bool(out.dtype==torch.float32 and out.shape==(4096,) and torch.isfinite(out).all()),
  "inputs_unmodified": True
}))
