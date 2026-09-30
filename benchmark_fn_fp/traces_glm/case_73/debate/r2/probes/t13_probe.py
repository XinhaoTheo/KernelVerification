import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/evidence_cases/case_e12/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
dev = "cuda" if torch.cuda.is_available() else "cpu"
smoke, W, b, c = km.make_inputs(dev)
bits = ((np.arange(4096)[:, None] >> np.arange(12)) & 1)
V = torch.from_numpy((2*bits.astype(np.float32) - 1)).to(dev)
Vd, Wd, Bd, Cd = V.double(), W.double(), b.double(), c.double()
residual = (torch.relu(Vd @ Wd.T - Bd) @ Cd)
smoke_err = float(torch.relu(smoke.double() @ Wd.T - Bd) @ Cd .max())
max_vertex_res = float(residual.max())
co_act = (torch.relu(Vd @ Wd.T - Bd) > 0).sum(dim=1)
print(json.dumps({
  "metric": "residual (omitted relu term) at all 4096 vertices vs the smoke-batch max residual; shows whether smoke rows miss worst case",
  "smoke_max_residual": smoke_err,
  "vertex_max_residual": max_vertex_res,
  "vertex_min_residual": float(residual.min()),
  "num_vertices_with_residual_gt_1": int((residual > 1.0).sum()),
  "max_coactivated_neurons": int(co_act.max()),
  "vertices_with_coactivation_ge_2": int((co_act >= 2).sum()),
  "smoke_covers_worst_case": bool(smoke_err >= max_vertex_res - 1e-12),
  "smoke_miss_factor": max_vertex_res / max(smoke_err, 1e-12)
}))
