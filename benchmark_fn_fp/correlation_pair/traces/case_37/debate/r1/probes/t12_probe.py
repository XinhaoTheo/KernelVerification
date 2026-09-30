
import json, importlib.util
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/cases/case_b/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

Es = []
outs = []
finite = True
for _ in range(5):
    x, A, B = k.make_inputs("cuda")
    o = k.run(x, A, B).detach().cpu().numpy()
    outs.append(o.copy())
    od = o.astype(np.float64)
    xd = x.cpu().numpy().astype(np.float64)
    ref = A.cpu().numpy().astype(np.float64) @ xd + B.cpu().numpy().astype(np.float64) @ xd
    den = max(np.linalg.norm(ref), 0.001*np.sqrt(64))
    Es.append(float(np.linalg.norm(od-ref)/den))
    finite = finite and bool(np.all(np.isfinite(od)))

bitwise_identical = all(np.array_equal(outs[0], o) for o in outs[1:])
res = dict(
  E_runs=Es, E_max=float(max(Es)), E_min=float(min(Es)),
  E_spread=float(max(Es)-min(Es)),
  tolerance=0.1, any_run_exceeds=bool(max(Es) > 0.1),
  margin_factor=float(0.1/max(Es)),
  bitwise_identical_across_runs=bitwise_identical,
  all_finite=finite, n_runs=5,
)
print(json.dumps(res))
