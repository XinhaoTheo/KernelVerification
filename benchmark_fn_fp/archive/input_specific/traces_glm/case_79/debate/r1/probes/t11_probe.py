import json
from fractions import Fraction
import numpy as np, torch

import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/evidence_cases/case_e18/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

x, = kern.make_inputs_numpy()  # stored fp32 array (host, CPU)
# exact real target per row using Fractions on the stored float32 values
target = [float(sum(Fraction(int(v.item()) ,1)*0 if False else Fraction(v.item()) for v in x[r])) for r in range(4)]
target = [float(sum(Fraction(v.item()) for v in x[r])) for r in range(4)]

xg, = kern.make_inputs()
before = xg.clone()
out = kern.run(xg)
actual = out.detach().cpu().numpy().astype(np.float64)
err = float(np.linalg.norm(actual - np.array(target)) / max(np.linalg.norm(np.array(target)), 1e-12))
unmodified = bool(torch.equal(xg, before))
structural = tuple(out.shape)==(4,) and out.dtype==torch.float32 and bool(np.isfinite(actual).all())
res = {"stored_row0": [float(v) for v in x[0]],
       "exact_target": target, "output": actual.tolist(),
       "relative_l2_error": err, "tolerance": 1e-5,
       "shape_ok": structural, "inputs_unmodified": unmodified,
       "passed_contract": bool(structural and unmodified and err <= 1e-5)}
print(json.dumps(res))