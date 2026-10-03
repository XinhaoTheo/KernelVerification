
import sys, json, importlib.util
sys.path.insert(0, '/root/pilot_cases/case_83')
spec = importlib.util.spec_from_file_location('k', '/root/pilot_cases/case_83/kernel.py')
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs()
out = k.run(a, b)
import numpy as np, torch
af = a.cpu().numpy().astype(np.float64)
bf = b.cpu().numpy().astype(np.float64)
ref = np.empty_like(af)
h = np.zeros(af.shape[1])
for t in range(af.shape[0]):
    h = af[t]*h + bf[t]
    ref[t] = h
o = out.cpu().numpy().astype(np.float64)
diff = o - ref
n = diff.size
E = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(n))
print(json.dumps({
  'metric': 'relative_l2_error_E',
  'E': float(E),
  'budget': 0.003,
  'finite': bool(np.isfinite(o).all()),
  'ref_norm': float(np.linalg.norm(ref.ravel())),
  'max_abs_err': float(np.abs(diff).max()),
  'max_rel_err': float((np.abs(diff)/np.maximum(np.abs(ref),1e-12)).max()),
  'shape': list(o.shape), 'dtype': str(out.dtype),
  'verdict_per_contract': bool(np.isfinite(o).all() and E <= 0.003)
}))
