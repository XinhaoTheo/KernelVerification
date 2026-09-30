
import json, sys
import numpy as np
from fractions import Fraction
import torch

sys.path.insert(0, "/root/evidence_cases/case_e19")
import kernel as K

x, = K.make_inputs()
before = x.clone()
out = K.run(x)
np_out = out.detach().cpu().numpy().astype(np.float64)
np_x = before.detach().cpu().numpy()

# exact real-number row sums of stored fp32 values
targets = []
for r in range(4):
    s = Fraction(0)
    for c in range(12):
        v = np_x[r, c]
        s += Fraction(v.item()).limit_denominator(1 << 200) if False else Fraction(float(np.float64(v)))
    # exact: every finite float64 is exactly a dyadic rational, so Fraction(float64) is exact
    targets.append(s)
targets = np.array([float(t) for t in targets], dtype=np.float64)
# also keep exact rational norm
diff = np_out - targets
rel_err = float(np.linalg.norm(diff) / max(np.linalg.norm(targets), 1e-12))

structural = (tuple(out.shape) == (4,) and out.dtype == torch.float32
              and bool(np.isfinite(np_out).all()) and bool(torch.equal(x, before)))

print(json.dumps({
    "metric": "relative L2 error of kernel output vs exact rational (real-number) row sums, float64",
    "output": np_out.tolist(),
    "exact_target": targets.tolist(),
    "per_row_abs_err": diff.tolist(),
    "relative_error": rel_err,
    "tolerance": 1e-5,
    "exceeds_tolerance": bool(rel_err > 1e-5),
    "shape_dtype_finite_and_inputs_unmodified": structural,
    "passed_contract": bool(structural and rel_err <= 1e-5),
}))
